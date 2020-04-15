# PyChrono imports
import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.sensor as sens

from control_utilities import Track

def SetChronoDataDirectories():
    """
    Set data directory

    This is useful so data directory paths don't need to be changed everytime
    you pull from or push to github. To make this useful, make sure you perform
    step 2, as defined for your operating system.

    For Linux or Mac users:
      Replace bashrc with the shell your using. Could be .zshrc.
      1. echo 'export CHRONO_DATA_DIR=<chrono's data directory>' >> ~/.bashrc
          Ex. echo 'export CHRONO_DATA_DIR=/home/user/chrono/data/' >> ~/.zshrc
      2. source ~/.zshrc

    For Windows users:
      Link as reference: https://helpdeskgeek.com/how-to/create-custom-environment-variables-in-windows/
      1. Open the System Properties dialog, click on Advanced and then Environment Variables
      2. Under User variables, click New... and create a variable as described below
          Variable name: CHRONO_DATA_DIR
          Variable value: <chrono's data directory>
              Ex. Variable value: C:\ Users\ user\ chrono\ data\
    """
    from pathlib import Path
    import os

    CONDA_PREFIX = os.environ.get('CONDA_PREFIX')
    CHRONO_DATA_DIR = os.environ.get('CHRONO_DATA_DIR')
    if CONDA_PREFIX and not CHRONO_DATA_DIR:
        CHRONO_DATA_DIR = os.path.join(CONDA_PREFIX, 'share', 'chrono', 'data', '')
    if not CHRONO_DATA_DIR:
        CHRONO_DATA_DIR = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parents[1], 'chrono', 'data', '')
    elif not CHRONO_DATA_DIR:
        raise Exception('Cannot find the chrono data directory. Please verify that CHRONO_DATA_DIR is set correctly.')

    chrono.SetChronoDataPath(CHRONO_DATA_DIR)
    veh.SetDataPath(os.path.join(CHRONO_DATA_DIR, 'vehicle', ''))

def CalcInitialPose(p1 : chrono.ChVectorD, p2 : chrono.ChVectorD, z=0.1, reversed=0):
    if not isinstance(p1, chrono.ChVectorD):
        raise TypeError
    elif not isinstance(p2, chrono.ChVectorD):
        raise TypeError

    p1.z = p2.z = z

    initLoc = p1

    vec = p2 - p1
    theta = math.atan2((vec%chrono.ChVectorD(1,0,0)).Length(),vec^chrono.ChVectorD(1,0,0))
    if reversed:
        theta *= -1
    initRot = chrono.ChQuaternionD()
    initRot.Q_from_AngZ(theta)

    return initLoc, initRot

def GenerateHallwayTrack(z=.15, width=1.1):
    points = [[-8.713, -1.646],
              [-7.851, -1.589],
              [-6.847, -1.405],
              [-6.048, -1.449],
              [-5.350, -1.658],
              [-4.628, -1.767],
              [-3.807, -1.789],
              [-2.865, -1.778],
              [-1.823, -1.743],
              [-0.724, -1.691],
              [0.373, -1.650],
              [1.411, -1.527],
              [2.349, -1.453],
              [3.174, -1.439],
              [3.915, -1.474],
              [4.652, -1.513],
              [5.487, -1.694],
              [6.506, -1.756],
              [7.506, -1.456],
              [7.732, -1.060],
              [7.983, -0.617],
              [7.432, 1.112],
              [6.610, 1.143],
              [5.688, 1.206],
              [4.950, 1.281],
              [4.331, 1.337],
              [3.754, 1.349],
              [3.152, 1.303],
              [2.478, 1.207],
              [1.708, 1.077],
              [0.832, 0.940],
              [-0.143, 0.828],
              [-1.201, 0.767],
              [-2.318, 0.781],
              [-3.463, 0.830],
              [-4.605, 0.838],
              [-5.715, 0.864],
              [-6.765, 0.934],
              [-7.737, 1.121],
              [-8.822, 1.318],
              [-10.024, 0.608],
              [-10.102, 0.437],
              [-10.211, -0.569],
              [-9.522, -1.514],
              [-8.713, -1.646]]
    track = Track(points, width=width, z=z)
    track.generateTrack(z=z)
    return track
