

def cents2pitchbend(cents, maxdeviation=200):
    """ 
    cents: an integer between -maxdeviation and +maxdviation
    """
    return int((cents + maxdeviation) / (maxdeviation * 2.0) * 16383.0 + 0.5)


def pitchbend2cents(pitchbend, maxcents=200):
    return int(((pitchbend / 16383.0) * (maxcents * 2.0)) - maxcents + 0.5)


def fileinfo(path):
    import midi
    m = midi.read_midifile(path)
    resolution = float(m.resolution)

    def tick2sec(tick, tempo):
        beats = tick / resolution
        secs = beats * (60 / tempo)
        return secs

    def trackdur(trackidx):
        tempo = 120
        now = 0
        for ev in m[trackidx]:
            dt = tick2sec(ev.tick, tempo) 
            now += dt
            if ev.name == "Set Tempo":
                tempo = int(ev.get_bpm() + 0.5)
        return now

    def tracktempos(trackidx):
        tempos = []
        for ev in m[trackidx]:
            if ev.name == "Set Tempo":
                bpm = int(ev.get_bpm() + 0.5)
                tempos.append(bpm)
        return tempos

    dur = max(trackdur(tracknum) for tracknum in range(len(m)))
    tempostr = ":".join(map(str, tracktempos(0)))

    info = {
        'dur': dur,
        'numtracks': len(m),
        'tickmode': 'relative' if m.tick_relative else 'absolute',
        'resolution': int(resolution),
        'tempo': tempostr
    }

    return info


def fileinfo_show(path):
    info = fileinfo(path)
    maxwidth = max(len(key) for key in info)
    for key, value in info.items():
        print(key.ljust(maxwidth), value)
