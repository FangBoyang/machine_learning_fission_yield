# 配置继承树 (CONFIG_TREE)

> 自动生成（pipeline/dump_resolved.py）。每个变体的**全展开参数**见
> `configs/resolved/<variant>.yaml`；本文件只描述**继承关系**。
> 源 config 仍用 `inherit` 作唯一真源，请勿手改 resolved/ 下文件。

## 继承树（root → leaf）

```
base
├── g_rawY_delta_np
│   ├── g_logY_delta_np
│   ├── h_rawY_noDelta
│   └── j_power_delta_np
│       ├── j_cosinehold_delta_np
│       │   └── k_gridrefine_lbfgs_delta_np
│       │       └── l_narrow_p02_gridrefine_lbfgs_delta_np
│       ├── j_cosinesmooth_delta_np
│       ├── m_ft_235UALL_power_delta_np
│       ├── n_power_delta_np
│       │   ├── p_gef_isomer_delta_np
│       │   ├── q_gef_isomer_delta_np
│       │   ├── r_gef_isomer_delta_np
│       │   └── s_gef_isomer_delta_np
│       ├── o_ft_235UALL_power_delta_np
│       │   ├── o2_ft_235UALL_lr1e4_freeze_pat50
│       │   ├── o3_ft_235UALL_lr5e4_hold50_pat50
│       │   │   └── o4_ft_235UALL_lr5e4_hold80_wd5e4
│       │   └── o5_ft_235UALL_lr1e3_hold30_wd5e4_pat50
│       │       └── o6_ft_235UALL_lr1e3_hold50_wd5e4_pat50
│       └── p_ft_235UALL_power_delta_np
└── g_rawY_delta_np_smaller
    └── i_rawY_delta_np_smaller
        ├── i_rawY_delta_np_smaller_more_patient
        └── m_ft_235UALL_delta_np_smaller
```

## 变体 → 继承链 → 全展开文件

| 变体 | 继承链 (root→leaf) | 全展开文件 |
|------|--------------------|------------|
| base | base | resolved/base.yaml |
| g_logY_delta_np | base → g_rawY_delta_np → g_logY_delta_np | resolved/g_logY_delta_np.yaml |
| g_rawY_delta_np | base → g_rawY_delta_np | resolved/g_rawY_delta_np.yaml |
| g_rawY_delta_np_smaller | base → g_rawY_delta_np_smaller | resolved/g_rawY_delta_np_smaller.yaml |
| h_rawY_noDelta | base → g_rawY_delta_np → h_rawY_noDelta | resolved/h_rawY_noDelta.yaml |
| i_rawY_delta_np_smaller | base → g_rawY_delta_np_smaller → i_rawY_delta_np_smaller | resolved/i_rawY_delta_np_smaller.yaml |
| i_rawY_delta_np_smaller_more_patient | base → g_rawY_delta_np_smaller → i_rawY_delta_np_smaller → i_rawY_delta_np_smaller_more_patient | resolved/i_rawY_delta_np_smaller_more_patient.yaml |
| j_cosinehold_delta_np | base → g_rawY_delta_np → j_power_delta_np → j_cosinehold_delta_np | resolved/j_cosinehold_delta_np.yaml |
| j_cosinesmooth_delta_np | base → g_rawY_delta_np → j_power_delta_np → j_cosinesmooth_delta_np | resolved/j_cosinesmooth_delta_np.yaml |
| j_power_delta_np | base → g_rawY_delta_np → j_power_delta_np | resolved/j_power_delta_np.yaml |
| k_gridrefine_lbfgs_delta_np | base → g_rawY_delta_np → j_power_delta_np → j_cosinehold_delta_np → k_gridrefine_lbfgs_delta_np | resolved/k_gridrefine_lbfgs_delta_np.yaml |
| l_narrow_p02_gridrefine_lbfgs_delta_np | base → g_rawY_delta_np → j_power_delta_np → j_cosinehold_delta_np → k_gridrefine_lbfgs_delta_np → l_narrow_p02_gridrefine_lbfgs_delta_np | resolved/l_narrow_p02_gridrefine_lbfgs_delta_np.yaml |
| m_ft_235UALL_delta_np_smaller | base → g_rawY_delta_np_smaller → i_rawY_delta_np_smaller → m_ft_235UALL_delta_np_smaller | resolved/m_ft_235UALL_delta_np_smaller.yaml |
| m_ft_235UALL_power_delta_np | base → g_rawY_delta_np → j_power_delta_np → m_ft_235UALL_power_delta_np | resolved/m_ft_235UALL_power_delta_np.yaml |
| n_power_delta_np | base → g_rawY_delta_np → j_power_delta_np → n_power_delta_np | resolved/n_power_delta_np.yaml |
| o2_ft_235UALL_lr1e4_freeze_pat50 | base → g_rawY_delta_np → j_power_delta_np → o_ft_235UALL_power_delta_np → o2_ft_235UALL_lr1e4_freeze_pat50 | resolved/o2_ft_235UALL_lr1e4_freeze_pat50.yaml |
| o3_ft_235UALL_lr5e4_hold50_pat50 | base → g_rawY_delta_np → j_power_delta_np → o_ft_235UALL_power_delta_np → o3_ft_235UALL_lr5e4_hold50_pat50 | resolved/o3_ft_235UALL_lr5e4_hold50_pat50.yaml |
| o4_ft_235UALL_lr5e4_hold80_wd5e4 | base → g_rawY_delta_np → j_power_delta_np → o_ft_235UALL_power_delta_np → o3_ft_235UALL_lr5e4_hold50_pat50 → o4_ft_235UALL_lr5e4_hold80_wd5e4 | resolved/o4_ft_235UALL_lr5e4_hold80_wd5e4.yaml |
| o5_ft_235UALL_lr1e3_hold30_wd5e4_pat50 | base → g_rawY_delta_np → j_power_delta_np → o_ft_235UALL_power_delta_np → o5_ft_235UALL_lr1e3_hold30_wd5e4_pat50 | resolved/o5_ft_235UALL_lr1e3_hold30_wd5e4_pat50.yaml |
| o6_ft_235UALL_lr1e3_hold50_wd5e4_pat50 | base → g_rawY_delta_np → j_power_delta_np → o_ft_235UALL_power_delta_np → o5_ft_235UALL_lr1e3_hold30_wd5e4_pat50 → o6_ft_235UALL_lr1e3_hold50_wd5e4_pat50 | resolved/o6_ft_235UALL_lr1e3_hold50_wd5e4_pat50.yaml |
| o_ft_235UALL_power_delta_np | base → g_rawY_delta_np → j_power_delta_np → o_ft_235UALL_power_delta_np | resolved/o_ft_235UALL_power_delta_np.yaml |
| p_ft_235UALL_power_delta_np | base → g_rawY_delta_np → j_power_delta_np → p_ft_235UALL_power_delta_np | resolved/p_ft_235UALL_power_delta_np.yaml |
| p_gef_isomer_delta_np | base → g_rawY_delta_np → j_power_delta_np → n_power_delta_np → p_gef_isomer_delta_np | resolved/p_gef_isomer_delta_np.yaml |
| q_gef_isomer_delta_np | base → g_rawY_delta_np → j_power_delta_np → n_power_delta_np → q_gef_isomer_delta_np | resolved/q_gef_isomer_delta_np.yaml |
| r_gef_isomer_delta_np | base → g_rawY_delta_np → j_power_delta_np → n_power_delta_np → r_gef_isomer_delta_np | resolved/r_gef_isomer_delta_np.yaml |
| s_gef_isomer_delta_np | base → g_rawY_delta_np → j_power_delta_np → n_power_delta_np → s_gef_isomer_delta_np | resolved/s_gef_isomer_delta_np.yaml |
