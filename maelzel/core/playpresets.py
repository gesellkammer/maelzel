from .presetutils import makePreset

builtinPresets = [
    makePreset('sin',
        'aout1 oscili a(kamp), mtof(lag(kpitch, 0.01))',
        descr="simplest sine wave",
        builtin=True),

    makePreset('tsin',
        "aout1 oscili a(kamp), mtof(lag(kpitch+ktransp, klag))",
        params=dict(ktransp=0, klag=0.1),
        descr="transposable sine wave",
        builtin=True),

    makePreset('tri',
        r'''
        kfreq = mtof:k(lag(kpitch, 0.08))
        aout1 = vco2(1, kfreq,  12) * a(kamp)
        ''',
        descr="simple triangle wave",
        builtin=True),

    makePreset('ttri',
        r'''
        kfreq = mtof:k(lag(kpitch + ktransp, klag))
        aout1 = vco2(1, kfreq,  12) * a(kamp)
        if kfreqratio > 0 then
           aout1 = K35_lpf(aout1, kfreq*kfreqratio, kQ)
        endif
        ''',
        params=dict(ktransp=0, klag=0.1, kfreqratio=0, kQ=3),
        descr="transposable triangle wave with optional lowpass-filter",
        builtin=True),

    makePreset('saw',
        r'''
        kfreq = mtof:k(lag(kpitch, 0.01))
        aout1 = vco2(1, kfreq, 0) * a(kamp)
        ''',
        descr="simple saw-tooth",
        builtin=True),

    makePreset('tsaw',
        r'''
        kfreq = mtof:k(lag(kpitch + ktransp, klag))
        aout1 = vco2(1, kfreq, 0) * a(kamp)
        if kfreqratio > 0 then
          aout1 = K35_lpf(aout1, kfreq*kfreqratio, kQ)
        endif
        ''',
        params = dict(ktransp=0, klag=0.1, kfreqratio=0, kQ=3),
        descr="transposable saw with optional low-pass filtering",
        builtin=True),

    makePreset('tsqr',
        r'''
        aout1 = vco2(1, mtof(lag(kpitch+ktransp, klag), 10) * a(kamp)
        if kcutoff > 0 then
          aout1 moogladder aout1, port(kcutoff, 0.05), kresonance
        endif          
        ''',
        params=dict(ktransp=0, klag=0.1, kcutoff=0, kresonance=0.2),
        descr="square wave with optional filtering",
        builtin=True),

    makePreset('tpulse',
        r"aout1 vco2 kamp, mtof:k(lag:k(kpitch+ktransp, klag), 2, kpwm",
        params=dict(ktransp=0, klag=0.1, kpwm=0.5),
        descr="transposable pulse with moulatable pwm",
        builtin=True)
]


soundfontGeneralMidiInstruments = {
    'piano': 147,
    'clarinet': 61,
    'oboe': 58,
    'flute': 42,
    'violin': 47,
    'reedorgan': 52
}
