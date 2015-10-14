"""Maps action tuples to the action values they correspond to.

This relies on the ALE action enum. See the src/common/Constants.h
file from the ALE Github.

xyf signifies the 3-long tuple of (x action, y action, fire action)
"""

X_DIM = 3 # (noop, left, right)
Y_DIM = 3 # (noop, up, down)
FIRE_DIM = 2 # (noop, fire)

ACTION_SHAPE = (X_DIM, Y_DIM, FIRE_DIM)

ACTION_TO_XYF = [
    # noop, fire
    (0,0,0),
    (0,0,1),

    # 8 directions
    (0,1,0),
    (2,0,0),
    (1,0,0),
    (0,2,0),
    (2,1,0),
    (1,1,0),
    (2,2,0),
    (1,2,0),

    # 8 directions with fire
    (0,1,1),
    (2,0,1),
    (1,0,1),
    (0,2,1),
    (2,1,1),
    (1,1,1),
    (2,2,1),
    (1,2,1),
]

XYF_TO_ACTION = dict(
    zip(ACTION_TO_XYF, range(18))
)

