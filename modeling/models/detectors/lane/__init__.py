from .cond_lstr_2d import CondLSTR2D
from .cond_lstr_3d import CondLSTR3D
from .cond_lstr_vote_3d import CondLSTRVote3D


__factory = {
    'CondLSTR2D': CondLSTR2D,
    'CondLSTR3D': CondLSTR3D,
    'CondLSTRVote3D': CondLSTRVote3D,
}
