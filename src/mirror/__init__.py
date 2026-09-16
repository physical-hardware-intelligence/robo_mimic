"""mirror -- webcam hand-pose teleoperation for the SO-101.

Pipeline, left to right. Everything between the two ends is a pure function,
which is the whole reason the suite can run in CI with no camera and no arm:

    camera -> landmarks -> hand frame -> retarget -> project -> safety -> IK -> sink
   [impure]  [mediapipe]    [pure]       [pure]     [pure]    [pure]   [pure] [impure]

Two facts shape every design decision here; both are written up in docs/adr/:

1. MediaPipe reports hand-RELATIVE geometry only. `hand_world_landmarks` are
   metres about the hand's own geometric centre, and normalized `z` is relative
   to the wrist. Absolute hand position in the camera frame is not recoverable
   from one RGB camera, so orientation maps absolutely and position does not.
   -> docs/adr/002-incremental-position-clutch.md

2. The SO-101 has five degrees of freedom, not six. A hand pose has six. The
   discarded one is tool YAW, and it is surfaced rather than silently dropped.
   -> docs/adr/001-five-dof-projection.md
"""

__version__ = "0.0.1"
