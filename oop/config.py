import operator as op

def getcutstr(conf_cut):
    '''
    Examples: 
    "mtg_medva_==_4", "mtg_medva_==_7", "mtg_medva_==_3", "mtg_medva_==_5", "mtg_medva_==_6", "mtg_medva_>_0"
    '''

    if conf_cut:
        try:
            l_conf_cut = conf_cut.split("_")
            l_conf_cut[0] = l_conf_cut[0].upper()
            l_conf_cut[1] = VAR_MAP[l_conf_cut[1]]
            if l_conf_cut[2] not in OP_MAP:
                raise ValueError(f"Invalid cutstr. Third element of string must correpond to an operator. Got {l_conf_cut[2]}.")
            cut_str = " ".join(l_conf_cut)
        except:
            cut_str = ""
    else:
        cut_str = ""
    return cut_str

def splitcutstr(cutstr):
    if cutstr:
        sat, variable, operator, value = (cutstr.split(" ")[i] for i in range(4))
    else: 
        sat, variable, operator, value = ("" for i in range(4))
    return (sat, variable, operator, value)

def getbmthresholds(name):
    if name == "Unfiltered":
        aa = -0.4
        bb = -0.4
        c = 2.5
    elif name == "Low_Lat":
        aa = -0.9
        bb = 0.0
        c = 2.3
    elif name == "High_Zenith":
        aa = -1.0
        bb = 0.0
        c = 2.3
    elif name == "SH_Arid":
        aa = -1.0
        bb = 0.0
        c = 1.6
    elif name == "NH_Arid":
        aa = -1.0
        bb = 0.0
        c = 1.3
    elif name in ["SO2_Proxy", "Ash_Proxy"]:
        aa = -0.9
        bb = 0.0
        c = 2.3
    else:
        raise ValueError("Got a beta mask which isn't unfiltered or low_lat. Exiting...")

    return (aa, bb, c)

# ...existing code...
OP_MAP = {'==': op.eq, '!=': op.ne, '>': op.gt, '<': op.lt, '>=': op.ge, '<=': op.le}
OP_MAP_STR = {'==':'eq','!=':'ne','>':'gt','<':'lt','>=':'ge','<=':'le'}
VAR_MAP = {'medva': 'Median_VA_Confidence'}

SO2_cloud_latmin = 13.
SO2_cloud_latmax = 14.
SO2_cloud_lonmin = 40.
SO2_cloud_lonmax = 45.

codes_to_ignore = ["noret"]
RETRIEVAL_CODE_LABELS = {
    "conf7_c1": "MTG Conf 7, MSG Conf 0 fails: C1 Threshold",
    "conf7_other": "MTG Conf 7, MSG Conf 0 fails: Other",
    "conf4_conmask": "MTG Conf 4, MSG Conf 0 fails: Conservative Mask",
    "conf4_c4": "MTG Conf 4, MSG Conf 0 fails: C4 Threshold",
    "conf4_c4_conmask": "MTG Conf 4, MSG Conf 0 fails: C4 & Con Mask",
    "conf4_other": "MTG Conf 4, MSG Conf 0 fails: Other",
    "conf3_btdcutoff_btd3_conmask": "MTG Conf 3, MSG Conf 0 fails: BTDcutoff, BTD3, Con Mask",
    "conf3_btdcutoff_btd3": "MTG Conf 3, MSG Conf 0 fails: BTDcutoff, BTD3",
    "conf3_btdcutoff_conmask": "MTG Conf 3, MSG Conf 0 fails: BTDcutoff, Con Mask",
    "conf3_c3_btd3_conmask": "MTG Conf 3, MSG Conf 0 fails: C3, BTD3, Con Mask",
    "conf3_c3_btd3": "MTG Conf 3, MSG Conf 0 fails: C3, BTD3",
    "conf3_c3_conmask": "MTG Conf 3, MSG Conf 0 fails: C3, Con Mask",
    "conf3_btd3_conmask": "MTG Conf 3, MSG Conf 0 fails: BTD3, Con Mask",
    "conf3_btd3": "MTG Conf 3, MSG Conf 0 fails: BTD3",
    "conf3_conmask": "MTG Conf 3, MSG Conf 0 fails: Con Mask",
    "conf3_btdcutoff": "MTG Conf 3, MSG Conf 0 fails: BTDcutoff",
    "conf3_c3": "MTG Conf 3, MSG Conf 0 fails: C3",
    "conf3_other": "MTG Conf 3, MSG Conf 0 fails: Other",
    "conf1_c3_libmask": "MTG Conf 1, MSG Conf 0 fails: C3, Liberal Mask",
    "conf1_c4_libmask": "MTG Conf 1, MSG Conf 0 fails: C4, Liberal Mask",
    "conf1_c4": "MTG Conf 1, MSG Conf 0 fails: C4 Threshold",
    "conf1_c3": "MTG Conf 1, MSG Conf 0 fails: C3 Threshold",
    "conf1_libmask": "MTG Conf 1, MSG Conf 0 fails: Liberal Mask",
    "conf1_other": "MTG Conf 1, MSG Conf 0 fails: Other",
    "conf7_c1_msgconf1": "MTG Conf 7, MSG Conf 1 fails: C1 Threshold",
    "conf7_other_msgconf1": "MTG Conf 7, MSG Conf 1 fails: Other",
    "conf4_conmask_msgconf1": "MTG Conf 4, MSG Conf 1 fails: Conservative Mask",
    "conf4_c4_msgconf1": "MTG Conf 4, MSG Conf 1 fails: C4 Threshold",
    "conf4_c4_conmask_msgconf1": "MTG Conf 4, MSG Conf 1 fails: C4 & Con Mask",
    "conf4_other_msgconf1": "MTG Conf 4, MSG Conf 1 fails: Other",
    "conf3_btdcutoff_btd3_conmask_msgconf1": "MTG Conf 3, MSG Conf 1 fails: BTDcutoff, BTD3, Con Mask",
    "conf3_btdcutoff_btd3_msgconf1": "MTG Conf 3, MSG Conf 1 fails: BTDcutoff, BTD3",
    "conf3_btdcutoff_conmask_msgconf1": "MTG Conf 3, MSG Conf 1 fails: BTDcutoff, Con Mask",
    "conf3_c3_btd3_conmask_msgconf1": "MTG Conf 3, MSG Conf 1 fails: C3, BTD3, Con Mask",
    "conf3_c3_btd3_msgconf1": "MTG Conf 3, MSG Conf 1 fails: C3, BTD3",
    "conf3_c3_conmask_msgconf1": "MTG Conf 3, MSG Conf 1 fails: C3, Con Mask",
    "conf3_btd3_conmask_msgconf1": "MTG Conf 3, MSG Conf 1 fails: BTD3, Con Mask",
    "conf3_btd3_msgconf1": "MTG Conf 3, MSG Conf 1 fails: BTD3",
    "conf3_conmask_msgconf1": "MTG Conf 3, MSG Conf 1 fails: Con Mask",
    "conf3_btdcutoff_msgconf1": "MTG Conf 3, MSG Conf 1 fails: BTDcutoff",
    "conf3_c3_msgconf1": "MTG Conf 3, MSG Conf 1 fails: C3",
    "conf3_other_msgconf1": "MTG Conf 3, MSG Conf 1 fails: Other",
    "conf1_msgconf1": "MTG Conf 1, MSG Conf 1",
    "conf1_msgconf4": "MTG Conf 1, MSG Conf 4",
    "conf4_msgconf4": "MTG Conf 4, MSG Conf 4",
    "conf7_msgconf7": "MTG Conf 7, MSG Conf 7",
    "conf3_msgconf3": "MTG Conf 3, MSG Conf 3",
    "conf3_msgconf4": "MTG Conf 3, MSG Conf 4",
    "conf6_msgconf3": "MTG Conf 6, MSG Conf 3",
    "conf2_msgconf4": "MSG Conf 4, MTG Conf 2",
    "conf4_c4_conmask_msgconf2": "MTG Conf 4, MSG Conf 2 fails: C4 & Con Mask",
    "other": "Other",
}

DICT_CUT_CONSTRAINT = {
    'Med_VA_4': [{'variables':["BTD2_conf"], 'operators':['<='], 'values':['df_c4']},
                 {'variables':["BMCon"], 'operators':['=='], 'values':['T']}, 
                 {'variables':["BTD2_conf", "BMCon"], 'operators':['<=', '=='], 'values':['df_c4','T']},
                 {'variables':["PreFilter_VA_Confidence"], 'operators':['=='], 'values':[4]},
                 {'variables':["PostFilter_VA_Confidence"], 'operators':['=='], 'values':[4]},
                 {'variables':["Median_VA_Confidence"], 'operators':['=='], 'values':[4]},
                 {'variables':["BTD2_conf", "BMCon"], 'operators':['<=', '=='], 'values':['df_c4','T'], 'plotonly':'msg'},
                 {'variables':["BTD2_conf", "BMCon"], 'operators':['<=', '=='], 'values':['df_c4','T'], 'plotonly':'mtg'},
    ],
    'Med_VA_3': [{'variables':["BTD2_conf"], 'operators':['<='], 'values':[-0.1]},
                 {'variables':["BTD2_conf"], 'operators':['>'], 'values':['df_c3']},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':['df_BTD3thresh']},
                 {'variables':["BMCon"], 'operators':['=='], 'values':['T']}, 
                 {'variables':["BTD2_conf","BTD2_conf"], 'operators':['>', '<='], 'values':['df_c3', -0.1]},
                 {'variables':["BTD2_conf","BTD2_conf","VolcanicAsh_BTD3"], 'operators':['>', '<=', '<='], 'values':['df_c3', -0.1, 'df_BTD3thresh']},
                 {'variables':["BTD2_conf", "BTD2_conf", "VolcanicAsh_BTD3", "BMCon"], 'operators':['>', '<=', '<=', '=='], 'values':['df_c3',-0.1,'df_BTD3thresh','T']},
                 {'variables':["PreFilter_VA_Confidence"], 'operators':['=='], 'values':[3]},
                 {'variables':["PostFilter_VA_Confidence"], 'operators':['=='], 'values':[3]},
                 {'variables':["Median_VA_Confidence"], 'operators':['=='], 'values':[3]},
                 {'variables':["BTD2_conf"], 'operators':['<='], 'values':[-0.1], 'plotonly':'msg'},
                 {'variables':["BTD2_conf"], 'operators':['<='], 'values':[-0.1], 'plotonly':'mtg'},
                 {'variables':["BTD2_conf"], 'operators':['>'], 'values':['df_c3'], 'plotonly':'msg'},
                 {'variables':["BTD2_conf"], 'operators':['>'], 'values':['df_c3'], 'plotonly':'mtg'},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':['df_BTD3thresh'], 'plotonly':'msg'},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':['df_BTD3thresh'], 'plotonly':'mtg'},
                 {'variables':["BMCon"], 'operators':['=='], 'values':['T'], 'plotonly':'msg'}, 
                 {'variables':["BMCon"], 'operators':['=='], 'values':['T'], 'plotonly':'mtg'}, 
                 {'variables':["BTD2_conf","BTD2_conf"], 'operators':['>', '<='], 'values':['df_c3', -0.1], 'plotonly':'msg'},
                 {'variables':["BTD2_conf","BTD2_conf"], 'operators':['>', '<='], 'values':['df_c3', -0.1], 'plotonly':'mtg'},
                 {'variables':["BTD2_conf","BTD2_conf","VolcanicAsh_BTD3"], 'operators':['>', '<=', '<='], 'values':['df_c3', -0.1, 'df_BTD3thresh'], 'plotonly':'msg'},
                 {'variables':["BTD2_conf","BTD2_conf","VolcanicAsh_BTD3"], 'operators':['>', '<=', '<='], 'values':['df_c3', -0.1, 'df_BTD3thresh'], 'plotonly':'mtg'},
                 {'variables':["BTD2_conf", "BTD2_conf", "VolcanicAsh_BTD3", "BMCon"], 'operators':['>', '<=', '<=', '=='], 'values':['df_c3',-0.1,'df_BTD3thresh','T'], 'plotonly':'msg'},
                 {'variables':["BTD2_conf", "BTD2_conf", "VolcanicAsh_BTD3", "BMCon"], 'operators':['>', '<=', '<=', '=='], 'values':['df_c3',-0.1,'df_BTD3thresh','T'], 'plotonly':'mtg'},
                 {'variables':["PreFilter_VA_Confidence"], 'operators':['=='], 'values':[3], 'plotonly':'msg'},
                 {'variables':["PreFilter_VA_Confidence"], 'operators':['=='], 'values':[3], 'plotonly':'mtg'},
                 {'variables':["PostFilter_VA_Confidence"], 'operators':['=='], 'values':[3], 'plotonly':'msg'},
                 {'variables':["PostFilter_VA_Confidence"], 'operators':['=='], 'values':[3], 'plotonly':'mtg'},
                 {'variables':["Median_VA_Confidence"], 'operators':['=='], 'values':[3], 'plotonly':'msg'},
                 {'variables':["Median_VA_Confidence"], 'operators':['=='], 'values':[3], 'plotonly':'mtg'}
    ],
    'Med_VA_5': [{'variables':["BTD2_conf"], 'operators':['<='], 'values':[-0.1]},
                 {'variables':["BTD2_conf"], 'operators':['>'], 'values':['df_c3']},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':['df_BTD3thresh']},
                 {'variables':["BMLib"], 'operators':['=='], 'values':['T']}, 
                 {'variables':["BTD2_conf","BTD2_conf"], 'operators':['>', '<='], 'values':['df_c3', -0.1]},
                 {'variables':["BTD2_conf","BTD2_conf","VolcanicAsh_BTD3"], 'operators':['>', '<=', '<='], 'values':['df_c3', -0.1, 'df_BTD3thresh']},
                 {'variables':["BTD2_conf", "BTD2_conf", "VolcanicAsh_BTD3", "BMLib"], 'operators':['>', '<=', '<=', '=='], 'values':['df_c3',-0.1,'df_BTD3thresh','T']},
                 {'variables':["PreFilter_VA_Confidence"], 'operators':['=='], 'values':[3]},
                 {'variables':["PostFilter_VA_Confidence"], 'operators':['=='], 'values':[3]},
                 {'variables':["Median_VA_Confidence"], 'operators':['=='], 'values':[3]},
                 {'variables':["BTD2_conf"], 'operators':['<='], 'values':[-0.1], 'plotonly':'msg'},
                 {'variables':["BTD2_conf"], 'operators':['<='], 'values':[-0.1], 'plotonly':'mtg'},
                 {'variables':["BTD2_conf"], 'operators':['>'], 'values':['df_c3'], 'plotonly':'msg'},
                 {'variables':["BTD2_conf"], 'operators':['>'], 'values':['df_c3'], 'plotonly':'mtg'},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':['df_BTD3thresh'], 'plotonly':'msg'},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':['df_BTD3thresh'], 'plotonly':'mtg'},
                 {'variables':["BMLib"], 'operators':['=='], 'values':['T'], 'plotonly':'msg'}, 
                 {'variables':["BMLib"], 'operators':['=='], 'values':['T'], 'plotonly':'mtg'}, 
                 {'variables':["BTD2_conf","BTD2_conf"], 'operators':['>', '<='], 'values':['df_c3', -0.1], 'plotonly':'msg'},
                 {'variables':["BTD2_conf","BTD2_conf"], 'operators':['>', '<='], 'values':['df_c3', -0.1], 'plotonly':'mtg'},
                 {'variables':["BTD2_conf","BTD2_conf","VolcanicAsh_BTD3"], 'operators':['>', '<=', '<='], 'values':['df_c3', -0.1, 'df_BTD3thresh'], 'plotonly':'msg'},
                 {'variables':["BTD2_conf","BTD2_conf","VolcanicAsh_BTD3"], 'operators':['>', '<=', '<='], 'values':['df_c3', -0.1, 'df_BTD3thresh'], 'plotonly':'mtg'},
                 {'variables':["BTD2_conf", "BTD2_conf", "VolcanicAsh_BTD3", "BMLib"], 'operators':['>', '<=', '<=', '=='], 'values':['df_c3',-0.1,'df_BTD3thresh','T'], 'plotonly':'msg'},
                 {'variables':["BTD2_conf", "BTD2_conf", "VolcanicAsh_BTD3", "BMLib"], 'operators':['>', '<=', '<=', '=='], 'values':['df_c3',-0.1,'df_BTD3thresh','T'], 'plotonly':'mtg'},
                 {'variables':["PreFilter_VA_Confidence"], 'operators':['=='], 'values':[3], 'plotonly':'msg'},
                 {'variables':["PreFilter_VA_Confidence"], 'operators':['=='], 'values':[3], 'plotonly':'mtg'},
                 {'variables':["PostFilter_VA_Confidence"], 'operators':['=='], 'values':[3], 'plotonly':'msg'},
                 {'variables':["PostFilter_VA_Confidence"], 'operators':['=='], 'values':[3], 'plotonly':'mtg'},
                 {'variables':["Median_VA_Confidence"], 'operators':['=='], 'values':[3], 'plotonly':'msg'},
                 {'variables':["Median_VA_Confidence"], 'operators':['=='], 'values':[3], 'plotonly':'mtg'}
    ],
    'Med_VA_6': [{'variables':["BTD2_conf"], 'operators':['<='], 'values':['df_c3']},
                 {'variables':["BTD2_conf"], 'operators':['>'], 'values':['df_c1']},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':['df_BTD3thresh']},
                 {'variables':["BMCon"], 'operators':['=='], 'values':['T']}, 
                 {'variables':["BTD2_conf","BTD2_conf"], 'operators':['>', '<='], 'values':['df_c1', 'df_c3']},
                 {'variables':["BTD2_conf","BTD2_conf","VolcanicAsh_BTD3"], 'operators':['>', '<=', '<='], 'values':['df_c1', 'df_c3', 'df_BTD3thresh']},
                 {'variables':["BTD2_conf", "BTD2_conf", "VolcanicAsh_BTD3", "BMCon"], 'operators':['>', '<=', '<=', '=='], 'values':['df_c1','df_c3','df_BTD3thresh','T']},
                 {'variables':["PreFilter_VA_Confidence"], 'operators':['=='], 'values':[6]},
                 {'variables':["PostFilter_VA_Confidence"], 'operators':['=='], 'values':[6]},
                 {'variables':["Median_VA_Confidence"], 'operators':['=='], 'values':[6]}
    ],
    'masked_test': [{'variables':["c3"], 'operators':['=='], 'values':['df_c3'], 'plotonly':'msg'}],
    'Med_VA_gt_0': [{'variables':["PreFilter_VA_Confidence"], 'operators':['>'], 'values':[0]},
                 {'variables':["PostFilter_VA_Confidence"], 'operators':['>'], 'values':[0]},
                 {'variables':["BTD2_conf", "VolcanicAsh_BTD3"], 'operators':['<=', 'or_<='], 'values':[-1, 1.5]},
                 {'variables':["BTD2_conf", "VolcanicAsh_BTD3"], 'operators':['<=', 'or_<='], 'values':[-1, 1.6]},
                 {'variables':["BTD2_conf", "VolcanicAsh_BTD3"], 'operators':['<=', 'or_<='], 'values':[-1, 1.7]},
                 {'variables':["BTD2_conf", "VolcanicAsh_BTD3"], 'operators':['<=', 'or_<='], 'values':[-1, 1.8]},
                 {'variables':["BTD2_conf", "VolcanicAsh_BTD3"], 'operators':['<=', 'or_<='], 'values':[-1, 1.9]},
                 {'variables':["BTD2_conf", "VolcanicAsh_BTD3"], 'operators':['<=', 'or_<='], 'values':[-1, 2.0]},
                 {'variables':["BTD2_conf", "VolcanicAsh_BTD3"], 'operators':['<=', 'or_<='], 'values':[-0.9, 1.6]},
                 {'variables':["BTD2_conf", "VolcanicAsh_BTD3"], 'operators':['<=', 'or_<='], 'values':[-0.8, 1.7]},
                 {'variables':["BTD2_conf", "VolcanicAsh_BTD3"], 'operators':['<=', 'or_<='], 'values':[-0.7, 1.8]},
                 {'variables':["BTD2_conf", "VolcanicAsh_BTD3"], 'operators':['<=', 'or_<='], 'values':[-0.6, 1.9]},
                 {'variables':["BTD2_conf", "VolcanicAsh_BTD3"], 'operators':['<=', 'or_<='], 'values':[-0.5, 2.0]},
                 {'variables':["BTD2_conf", "VolcanicAsh_BTD3"], 'operators':['<=', 'or_<='], 'values':[-0.4, 2.1]},
                 {'variables':["BTD2_conf", "VolcanicAsh_BTD3"], 'operators':['<=', 'or_<='], 'values':[-0.3, 2.2]},
                 {'variables':["BTD2_conf", "VolcanicAsh_BTD3"], 'operators':['<=', 'or_<='], 'values':[-0.2, 2.3]},
                 {'variables':["BTD2_conf", "VolcanicAsh_BTD3"], 'operators':['<=', 'or_<='], 'values':[-0.1, 2.4]},
                 {'variables':["BTD2_conf", "VolcanicAsh_BTD3"], 'operators':['<=', 'or_<='], 'values':[0., 2.5]},
                 {'variables':["BTD2_conf"], 'operators':['<='], 'values':[-1]},
                 {'variables':["BTD2_conf"], 'operators':['<='], 'values':[-1]},
                 {'variables':["BTD2_conf"], 'operators':['<='], 'values':[-1]},
                 {'variables':["BTD2_conf"], 'operators':['<='], 'values':[-1]},
                 {'variables':["BTD2_conf"], 'operators':['<='], 'values':[-1]},
                 {'variables':["BTD2_conf"], 'operators':['<='], 'values':[-1]},
                 {'variables':["BTD2_conf"], 'operators':['<='], 'values':[-0.9]},
                 {'variables':["BTD2_conf"], 'operators':['<='], 'values':[-0.8]},
                 {'variables':["BTD2_conf"], 'operators':['<='], 'values':[-0.7]},
                 {'variables':["BTD2_conf"], 'operators':['<='], 'values':[-0.6]},
                 {'variables':["BTD2_conf"], 'operators':['<='], 'values':[-0.5]},
                 {'variables':["BTD2_conf"], 'operators':['<='], 'values':[-0.4]},
                 {'variables':["BTD2_conf"], 'operators':['<='], 'values':[-0.3]},
                 {'variables':["BTD2_conf"], 'operators':['<='], 'values':[-0.2]},
                 {'variables':["BTD2_conf"], 'operators':['<='], 'values':[-0.1]},
                 {'variables':["BTD2_conf"], 'operators':['<='], 'values':[0.]},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':[1.5]},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':[1.6]},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':[1.7]},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':[1.8]},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':[1.9]},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':[2.0]},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':[1.6]},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':[1.7]},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':[1.8]},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':[1.9]},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':[2.0]},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':[2.1]},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':[2.2]},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':[2.3]},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':[2.4]},
                 {'variables':["VolcanicAsh_BTD3"], 'operators':['<='], 'values':[2.5]},
                 {'variables':["Median_VA_Confidence"], 'operators':['>'], 'values':[0]}
    ],
    'Med_VA_7': [{'variables':["BTD2_conf"], 'operators':['<='], 'values':['df_c1']},
                 {'variables':["PreFilter_VA_Confidence"], 'operators':['=='], 'values':[7]},
                 {'variables':["PostFilter_VA_Confidence"], 'operators':['=='], 'values':[7]},
                 {'variables':["Median_VA_Confidence"], 'operators':['=='], 'values':[7]}
    ]
}

def select_cmap(n):
    if n < 9: return plt.get_cmap('Dark2', n)
    if n < 11: return plt.get_cmap('tab10', n)
    if n < 13: return plt.get_cmap('Paired', n)
    return plt.get_cmap('tab20', n)
