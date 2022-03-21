import enum

class WindSource(enum.Enum):
    sounding = 0
    rap_130 = 1

def GetWindSourceDescription(in_enum):
    if in_enum == WindSource.sounding:
        return 'sounding'
    elif in_enum == WindSource.rap_130:
        return 'rap 130'