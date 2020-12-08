from typing import Optional, Tuple, List, Dict
from copy import deepcopy
import pandas as pd
import numpy as np
import torch as pt
from torch import Tensor, BoolTensor, nn
from torch.nn import init, functional as F

from ..smooth import ExpSmooth
from ..base import GlobalLocalModel, XSeriesAttention



class RBFAttention(XSeriesAttention):
    def _compute_qkv(self,
                     query: Tensor,
                     ref: Tensor,
                     local_est: Optional[Tensor],
                     query_time_feats: Optional[Tensor],
                     ref_time_feats: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if (query_time_feats is not None) and (ref_time_feats is not None):
            assert query_time_feats.size(2) == ref_time_feats.size(2) == self.d_feats
            assert query_time_feats.size(1) == query.size(1)
            assert ref_time_feats.size(1) == ref.size(1)
            query_time_feats = query_time_feats.transpose(1,2)
            ref_time_feats = ref_time_feats.transpose(1,2)
        else:
            assert self.d_feats == 0
        # [M, Q] -> [M, 1, Q]
        q_input = pt.cumsum(query, dim=1).unsqueeze(dim=1)
        # [M, 1, Q] -> [M, H, Q-C+1] -> [M, Q-C+1, H]
        q, q_offset, q_scale = self.minmax_conv1d(
            x=q_input,
            w=self.q_weight,
            b=self.q_bias,
        )
        q = q.transpose(1,2)
        self.q_scale = q_scale.transpose(1,2)
        
        if local_est is None:
            rs = [None] * self.n_rolls
        else:
            rs = []
            for r in pt.split(local_est, self.pred_size, dim=2):
                n, l, d = r.size()
                r = r / self.q_scale
                r = F.relu(r)
                r = r.unsqueeze(dim=2).expand(n,l,self.n_output,d)
                r = r.contiguous().view(n,l,-1)
                rs.append(r)
            
        # [N, K] -> [N, 1, K-P]
        k_input = pt.cumsum(ref, dim=1).unsqueeze(dim=1)
        k_input = k_input[..., :-self.pred_size]
        # [N, 1, K-P] -> [N, K-P-C+1, H]
        k, k_offset, k_scale = self.minmax_conv1d(
            x=k_input, 
            w=self.k_weight, 
            b=self.k_bias, 
            feats=ref_time_feats,
        )
        k = k.transpose(1,2)

        # [N, K] -> [N, 1, K-C]
        v_input = ref.unsqueeze(dim=1)
        v_input = v_input[..., self.cond_size:]
        # [N, 1, K-C] -> [N, K-P-C+1, H] 
        v, _, _ = self.minmax_conv1d(
            x=v_input, 
            w=self.v_weight,
            b=self.v_bias, 
            offset=pt.zeros_like(k_offset),
            scale=k_scale,
        )
        v = v.transpose(1,2)
        v = F.relu(v)
        
        m = None
        return q, k, v, m, rs

    @staticmethod
    def _rbf(a: Tensor, b: Tensor) -> Tensor:
        dot = pt.einsum('mqh,nkh->mqkn', a, b)
        a = a.pow(2).sum(dim=-1)
        a = a.view(*a.shape,1,1)
        b = b.pow(2).sum(dim=-1)
        b = b.view(1,1,*b.shape).transpose(2,3)
        return 2*dot - a - b

    @staticmethod
    def _dot(a: Tensor, b: Tensor) -> Tensor:
        dot = pt.einsum('mqh,nkh->mqkn', a, b)
        return dot
    
    def _compute_scores(self,
                        q: Tensor,
                        k: Tensor,
                        r: Optional[Tensor],
                        query_space_feats: Optional[Tensor],
                        ref_space_feats: Optional[Tensor],
                        attn_mask: Optional[BoolTensor],
    ):
        full_score = self._rbf(q, k)
        if (query_space_feats is not None) and (ref_space_feats is not None):
            query_space_feats = query_space_feats.unsqueeze(dim=1)
            ref_space_feats = ref_space_feats.unsuqeeze(dim=0)
            space_score = self._rbf(query_space_feats, ref_space_feats)
            full_score = full_score + space_score
        if attn_mask is not None:
            full_score = full_score.masked_fill(attn_mask, float('-inf'))
        self.full_score = full_score
        
        if r is None:
            r_score = None
        else:
            r_score = -pt.sum((q - (self.r_key if self.fix_ar_key else q)) ** 2, dim=-1, keepdim=True)
            
        if self.add_autoreg:
            m_score = -pt.sum((q - (self.m_key if self.fix_ar_key else q)) ** 2, dim=-1, keepdim=True)
        else:
            m_score = None
        
        return full_score, r_score, m_score

    def semi_hard_attention(self,
                            full_score: Tensor,            # [m, q, k, n]
                            v: Tensor,                     # [n, k, h]
                            r_score: Optional[Tensor],     # [m, q, 1]
                            r: Optional[Tensor],           # [m, q, h]
                            m_score: Optional[Tensor],     # [m, q, 1]
                            m: Optional[Tensor],           # [m, q, h]
    ) -> Tensor:
        # take max by key time index
        # [m, q, k, n] -> [m, q, n]
        ref_score, max_idx_per_ref = full_score.max(dim=2)
        self.selected_t = max_idx_per_ref.detach() + self.cond_size
        # [m, q, n] -> [n, mq]
        v_idx = max_idx_per_ref.view(-1, max_idx_per_ref.size(-1)).transpose(0, 1)
        # [n, mq] -> [n, mq, h]
        v_idx = v_idx.unsqueeze(dim=-1).repeat(1,1,self.d_hidden)
        # [n, k, h] -> [n, mq, h] -> [m, q, n, h]
        selected_v = pt.gather(v, 1, v_idx).transpose(0, 1).view(
            *full_score.shape[:2],
            v.shape[0],
            self.d_hidden,
        )
        # [m, q, n, o]
        selected_v = F.linear(selected_v, self.o_weight, self.o_bias)
        selected_v = F.relu(selected_v)
        self.selected_v = selected_v
        # if r is not None:
        #     r = F.linear(r, self.o_weight, self.o_bias)
        # if m is not None:
        #     m = F.linear(m, self.o_weight, self.o_bias)

        if r_score is not None:
            ref_score = pt.cat([ref_score, r_score], dim=-1)
        if m_score is not None:
            ref_score = pt.cat([ref_score, m_score], dim=-1)
        ref_score = ref_score.div(self.temperature).softmax(dim=-1)
        ref_score = pt.masked_fill(ref_score, pt.isnan(ref_score), 0.0)
        self.ref_score = ref_score
        # [m, q, o]
        bias = 0.0
        if m_score is not None:
            size = ref_score.size(-1)
            ref_score, m_score = ref_score.split([size-1, 1], dim=-1)
            bias = bias + m_score * m
        if r_score is not None:
            size = ref_score.size(-1)
            ref_score, r_score = ref_score.split([size-1, 1], dim=-1)
            bias = bias + r_score * r
        p = ref_score.unsqueeze(dim=-1).mul(selected_v).sum(dim=-2)
        p = p + bias
        return p
    
    def forward(self, 
                query: Tensor, 
                ref: Tensor, 
                local_est: Optional[Tensor] = None,
                query_space_feats: Optional[Tensor] = None,
                ref_space_feats: Optional[Tensor] = None,
                query_time_feats: Optional[Tensor] = None,
                ref_time_feats: Optional[Tensor] = None,
                attn_mask: Optional[BoolTensor] = None,
    ):
        q, k, v, m, rs = self._compute_qkv(
            query, ref, local_est, 
            query_time_feats, ref_time_feats,
        )
        
        full_score, r_score, m_score = self._compute_scores(
            q, k, rs[0], query_space_feats, ref_space_feats, attn_mask,
        )
            
        preds = []
        for skip in range(self.n_rolls):
            skip_size = self.pred_size * skip
            score = full_score[..., :(full_score.size(2)-skip_size), :]
            value = v[:, skip_size:]
            # [M, Q, E, R]
            if self.full_attention:
                pr = self.fully_soft_attention(
                    score, value, r_score, rs[skip], m_score, m,
                )
            else:
                pr = self.semi_hard_attention(
                    score, value, r_score, rs[skip], m_score, m,
                )
            if self.cum_value:
                pr = pr.unsqueeze(dim=3)
            else:
                pr = pr.view(*pr.shape[:-1], self.n_output, self.pred_size)
            preds.append(pr)
        pr = pt.cat(preds, dim=3)
        pr = pr * self.q_scale.unsqueeze(dim=3)
        return pr
    


class CalHospsModel(GlobalLocalModel):
    def __init__(self,
                 cond_size: int,
                 pred_size: int,
                 n_rolls: int,
                 d_hidden: int,
                 n_location: int,
                 quantiles: List[int],
                 d_feats: int = 0,
                 share_params: bool = False,
                 full_attention: bool = False,
                 symmetric: bool = False,
                 var_penalty: float = 1.0,
    ) -> None:
        super(CalHospsModel, self).__init__(n_location, quantiles, share_params)

        self.cond_size = cond_size
        self.pred_size = pred_size
        self.n_rolls = n_rolls
        self.d_hidden = d_hidden
        self.d_feats = d_feats
        self.full_attention = full_attention
        self.symmetric = symmetric
        self.var_penalty = var_penalty
        
        self.cases_smoother = ExpSmooth(
            cond_size=cond_size,
            pred_size=pred_size*n_rolls,
            n_location=n_location,
            share_params=share_params,
        )
        self.hosps_smoother = ExpSmooth(
            cond_size=cond_size, 
            pred_size=pred_size*n_rolls, 
            n_location=n_location,
            share_params=share_params,
        )
        self.attention = RBFAttention(
            cond_size=cond_size,
            pred_size=pred_size,
            d_hidden=d_hidden,
            d_feats=d_feats,
            n_rolls=n_rolls,
            n_output=self.n_output,
            full_attention=full_attention,
            symmetric=symmetric,
            cum_value=False,
            add_autoreg=False,
            fix_ar_key=False,
            temperature=1.0,
        )

        self.register_buffer('smoothed_hosps', None, persistent=False)
        self.register_buffer('smoothed_cases', None, persistent=False)
        self.register_buffer('residual_hosps', None, persistent=False)
        self.register_buffer('residual_cases', None, persistent=False)
        self.register_buffer('smoothed_pr', None, persistent=False)
        self.register_buffer('residual_pr', None, persistent=False)
        self.register_buffer('hosps_local_pr', None, persistent=False)

    def forward(self, 
                hosps_data: Tensor,
                cases_data: Tensor,
                test_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Dict]:
        smoothed_hosps, hosps_local_pr = self.hosps_smoother(hosps_data)
        smoothed_cases, cases_local_pr = self.cases_smoother(cases_data)
        
        residual_hosps = hosps_data - smoothed_hosps
        residual_cases = cases_data - smoothed_cases

        self.smoothed_hosps = smoothed_hosps.detach()
        self.smoothed_cases = smoothed_cases.detach()
        self.residual_hosps = residual_hosps.detach()
        self.residual_cases = residual_cases.detach()
        
        hosps_length = hosps_data.size(1)
        attn_mask = pt.ones(
            hosps_length-self.cond_size+1,
            hosps_length-self.cond_size-self.pred_size+1,
            dtype=pt.bool, device=hosps_data.device,
        ).triu()
        attn_mask = attn_mask.view(1, *attn_mask.shape, 1)
        target_index = pt.tensor(
            np.arange(self.cond_size, hosps_length+1).reshape(-1,1)\
            + np.arange(self.pred_size * self.n_rolls).reshape(1,-1),
            dtype=pt.long, device=hosps_data.device,
        )
        target_mask = target_index >= hosps_length
        target_index = pt.where(target_mask, pt.zeros_like(target_index)-1, target_index)
        target = hosps_data[:, target_index]
        
        smoothed_query = smoothed_hosps
        smoothed_ref = pt.cat([smoothed_hosps, smoothed_cases], dim=0)
        residual_query = residual_hosps
        residual_ref = pt.cat([residual_hosps, residual_cases], dim=0)
        
        if test_size is not None:
            smoothed_query = smoothed_query[:, -(test_size+self.cond_size):]
            residual_query = residual_query[:, -(test_size+self.cond_size):]
            attn_mask = attn_mask[:, -(test_size+1):]
            target = target[:, -(test_size+1):]
            target_mask = target_mask[-(test_size+1):]
            hosps_local_pr = hosps_local_pr[:, -(test_size+1):]
        
        smoothed_pr = self.attention(
            query=smoothed_query,
            ref=smoothed_ref,
            local_est=hosps_local_pr,
            attn_mask=attn_mask,
        )

        # residual_pr = self.attention(
        #     query=residual_query,
        #     ref=residual_ref,
        #     local_est=None,
        #     attn_mask=attn_mask,
        # )
        
        self.smoothed_pr = smoothed_pr.detach()
        # self.residual_pr = residual_pr.detach()
        self.hosps_local_pr = hosps_local_pr.detach()

        # pr = pt.clamp_min(smoothed_pr + residual_pr, min=0.0)
        pr = pt.clamp_min(smoothed_pr, 0.0)
        loss = sum(
            self.quantile_error(p, target, q) 
            for q, p in zip(
                self.quantiles,
                pr.unbind(dim=2),
            )
        )
        loss = loss.masked_fill(target_mask, 0.0).mean()
        loss = loss + (self.cases_smoother.level_diffs.mean() + self.hosps_smoother.level_diffs.mean()) * self.var_penalty
        return loss, pr

