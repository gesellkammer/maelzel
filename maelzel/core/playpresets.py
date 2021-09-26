from .presetbase import PresetDef

builtinPresets = [
    PresetDef('sin',
        'aout1 oscili a(kamp), mtof(lag(kpitch, 0.01))',
        description="simplest sine wave",
        builtin=True),

    PresetDef('tsin',
        "aout1 oscili a(kamp), mtof(lag(kpitch+ktransp, klag))",
        params=dict(ktransp=0, klag=0.1),
        description="transposable sine wave",
        builtin=True),

    PresetDef('tri',
        r'''
        kfreq = mtof:k(lag(kpitch, 0.08))
        aout1 = vco2(1, kfreq,  12) * a(kamp)
        ''',
        description="simple triangle wave",
        builtin=True),

    PresetDef('ttri',
        r'''
        kfreq = mtof:k(lag(kpitch + ktransp, klag))
        aout1 = vco2(1, kfreq,  12) * a(kamp)
        if kfreqratio > 0 then
           aout1 = K35_lpf(aout1, kfreq*kfreqratio, kQ)
        endif
        ''',
        params=dict(ktransp=0, klag=0.1, kfreqratio=0, kQ=3),
        description="transposable triangle wave with optional lowpass-filter",
        builtin=True),

    PresetDef('saw',
        r'''
        kfreq = mtof:k(lag(kpitch, 0.01))
        aout1 = vco2(1, kfreq, 0) * a(kamp)
        ''',
        description="simple saw-tooth",
        builtin=True),

    PresetDef('tsaw',
        r'''
        kfreq = mtof:k(lag(kpitch + ktransp, klag))
        aout1 = vco2(1, kfreq, 0) * a(kamp)
        if kfreqratio > 0 then
          aout1 = K35_lpf(aout1, kfreq*kfreqratio, kQ)
        endif
        ''',
        params = dict(ktransp=0, klag=0.1, kfreqratio=0, kQ=3),
        description="transposable saw with optional low-pass filtering",
        builtin=True),

    PresetDef('tsqr',
        r'''
        aout1 = vco2(1, mtof(lag(kpitch+ktransp, klag), 10) * a(kamp)
        if kcutoff > 0 then
          aout1 moogladder aout1, port(kcutoff, 0.05), kresonance
        endif          
        ''',
        params=dict(ktransp=0, klag=0.1, kcutoff=0, kresonance=0.2),
        description="square wave with optional filtering",
        builtin=True),

    PresetDef('tpulse',
        r"aout1 vco2 kamp, mtof:k(lag:k(kpitch+ktransp, klag), 2, kpwm",
        params=dict(ktransp=0, klag=0.1, kpwm=0.5),
        description="transposable pulse with moulatable pwm",
        builtin=True)
]


soundfontGeneralMidiPresets = {
    '.piano': (0, 0),     # "Yamaha Grand Piano"
    '.clarinet': (0, 71),
    '.oboe': (0, 68),
    '.flute': (0, 73),
    '.violin': (0, 40),
    '.reedorgan': (0, 20)
}
