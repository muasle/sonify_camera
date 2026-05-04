from mmm_audio import *

struct Ping(PolyObject):
    var world: World  
    var osc: Osc[]
    var freq: Float64
    var env: Env

    var m: Messenger
    var trig: Bool

    var pan: Float64
    var vol: Float64

    fn check_active(mut self) -> Bool:
        return self.env.is_active

    fn set_trigger(mut self, trig: Bool):
        self.trig = trig

    fn __init__(out self, world: World):
        self.world = world
        
        self.osc = Osc(self.world)
        self.freq = 600.0

        self.env = Env(world)
        self.env.params.values = [0,1,0]
        self.env.params.times = [0.001, 0.1]

        self.m = Messenger(world, "position")
        self.trig = False

        self.pan = 0.0
        self.vol = 0.1

    fn next(mut self) -> MFloat[2]:        
        env = self.env.next(self.trig)

        osc = self.osc.next(self.freq, osc_type=OscType.sine)

        sig = osc * env * self.vol

        return pan2(sig, self.pan)

struct Sonify(Movable, Copyable):
    var world: World 
    var pings: List[Ping]
    var pings_poly: PolyTrigger

    fn __init__(out self, world: World):
        self.world = world
        self.pings = [Ping(world) for i in range(5)]
        self.pings_poly = PolyTrigger(5, 50, world, "ping", 5)

    fn next(mut self) -> MFloat[2]:
        fn call_back(mut voice: Ping, mut vals: List[Float64]):
            # print(vals)
            if len(vals) >= 1:
                voice.freq = vals[0]
            if len(vals) >= 2:
                voice.vol = vals[1]

        self.pings_poly.next(self.pings, call_back)

        out = MFloat[2](0.0)
        for i in range(len(self.pings)):
            out += self.pings[i].next()

        return out
