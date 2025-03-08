### script to collate analysis data from optoTiam1, optoITSN, and optoPLC movies

tiam_auto_movies = [
    ("migration50",),
    ("migration53",[8,11]),
    ("migration54",),
    ("migration56",),
    ("migration61",),
    ("migration64",),
    ("migration65",),
    # ("migration70",), #no automatic analysis for it - fix?
]

tiam_manual_movies = [ #all of these come with a $manual tag
    ("migration53",),
    ("migration50",),
    ("migration70",),
]


itsn_auto_movies = [ 
    ("migration44",),
    ("migration43",[11,12]),
    ("migration42",[6,7,9,10,11]),
    ("migration41",[1,6]),
    ("itsn1_better",),
    ("itsn2",)
]

itsn_manual_movies = [
    ("itsn1",),
    ("itsn2",),
]


def 


# plc_auto_movies = [
#     ("plcpeg3",),
#     ("plcpeg4",),
# ]