
class StringAttributeMeta(type):
    def __init__(cls, name, bases, attrs):
        for attr_name, attr_value in attrs.items():
            if isinstance(attr_value, str):
                setattr(cls, attr_name, cls.make_setter(attr_name))

    @staticmethod
    def make_setter(attr_name):
        def setter(self, value):
            if not isinstance(value, str):
                raise ValueError(f"{attr_name} must be a string")
            setattr(self, "_" + attr_name, value)
        return setter

#class MtzLabels(metaclass=StringAttributeMeta):
class MtzLabels():
    def __init__(self):
        self.Fon       = ""
        self.Foff      = ""
        self.SigFon    = ""
        self.SigFoff   = ""
        self.PhiC      = ""
        self.FDiff     = ""
        self.SigDiff   = ""

class DataInfo:
    def __init__(self):
        self.Meta = self.Meta()
        self.Maps = self.Maps()
        self.Xtal = self.Xtal()

    #class Meta(metaclass=StringAttributeMeta):
    class Meta():
        def __init__(self):
            self.Name = ""
            self.Path = ""
            self.Pdb  = ""

    class Maps:
        def __init__(self):
            self.MapRes        = ""
            self.MaskSpacing   = ""
            self.HighRes       = None
    
    class Xtal:
        def __init__(self):
            self.SpaceGroup    = ""
            self.Cell          = ""


    

class itTVData:
    def __init__(self):
        self.Data = self.Data()
        self.Labels = self.Labels()
        self.Meta = self.Meta()
        self.itStats = self.itStats()

    class Data:
        def __init__(self):
            self.RawData = ""
            self.itDiff  = ""

    #class Labels(metaclass=StringAttributeMeta):
    class Labels():
        def __init__(self):
            self.Fon = ""
            self.Foff = ""
            self.PhiC = ""
            self.FDiff = ""
            self.PhiDiff = ""

    class Meta:
        def __init__(self):
            self.flags = ""
            self.lvalue = ""

    class itStats:
        def __init__(self):
            self.deltaphi = ""
            self.deltaproj = ""
            self.entropy = ""
            self.zvec = ""
    
    #Labels = StringAttributeMeta("Labels", (Labels,), Labels.__dict__)

class TVMap:
    def __init__(self):
        self.Data = self.Data()
        self.Meta = self.Meta()
        self.Stats = self.Stats()
        
    class Data:
        def __init__(self):
            self.RawMtz  = ""
            self.MapData = ""
            self.TVMtz   = ""
        
    class Meta:
        def __init__(self):
            self.Lambda  = ""
            self.BestType   = None #an indicator of optimized nege, error, or None?

    class Stats:
        def __init__(self):
            self.DeltaF   = ""
            self.DeltaPhi = ""
            self.Entropy  = ""
            self.CVError  = ""


