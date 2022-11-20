from .presetdef import PresetDef

builtinPresets = [
    PresetDef(
        'simplesin',
        'aout1 oscili a(kamp), mtof(lag(kpitch, 0.01))',
        description="simplest sine wave",
        builtin=True),

    PresetDef(
        'sin',
        "aout1 oscili a(kamp), mtof(lag(kpitch+ktransp, klag))",
        args=dict(ktransp=0, klag=0.1),
        description="transposable sine wave",
        builtin=True),

    PresetDef(
        'tri',
        r'''
        |ktransp=0, klag=0.1, kfreqratio=0, kQ=3|
        kfreq = mtof:k(lag(kpitch + ktransp, klag))
        aout1 = vco2(1, kfreq,  12) * a(kamp)
        if kfreqratio > 0 then
           aout1 = K35_lpf(aout1, kfreq*kfreqratio, kQ)
        endif
        ''',
        description="transposable triangle wave with optional lowpass-filter",
        builtin=True),

    PresetDef(
        'saw',
        r'''
        |ktransp=0, klag=0.1, kfreqratio=0,kQ=3|
        kfreq = mtof:k(lag(kpitch + ktransp, klag))
        asig = vco2(1, kfreq, 0) * a(kamp)
        aout1 = kfreqratio == 0 ? asig : K35_lpf(asig, kfreq*kfreqratio, kQ)
        ''',
        description="transposable saw with optional low-pass filtering",
        builtin=True),

    PresetDef(
        'sqr',
        r'''
        |ktransp=0, klag=0.1, kcutoff=0, kresonance=0.2|
        aout1 = vco2(1, mtof(lag(kpitch+ktransp, klag), 10) * a(kamp)
        aout1 = kcutoff == 0 ? aout1 : moogladder aout1, lag(kcutoff, 0.1), kresonance
        ''',
        description="square wave with optional filtering",
        builtin=True),

    PresetDef(
        'pulse',
        r"aout1 vco2 kamp, mtof:k(lag:k(kpitch+ktransp, klag), 2, kpwm",
        args=dict(ktransp=0, klag=0.1, kpwm=0.5),
        description="transposable pulse with modulatable pwm",
        builtin=True),

    PresetDef(
        '_click',
        r"""
        |ktransp=24|
        aclickenv expseg db(-120), 0.01, 1, 0.1, db(-120)
        aout1 = oscili:a(aclickenv, mtof:k(kpitch+ktransp))
        """,
        description="Default preset used when rendering a click-track",
        builtin=True),

    PresetDef(
        '_playtable',
        audiogen=r"""
        |isndtab=0, istart=0, icompensatesr=1, kspeed=1, ixfade=-1|
        ; ixfade: crossfade time, if negative no looping
        iloop = ixfade >= 0 ? 1 : 0
        inumouts = ftchnls(isndtab)
        inumsamples = nsamp(isndtab)
        isr = ftsr(isndtab)
        
        if isr <= 0 then
            initerror sprintf("Could not determine sr of table %d", isndtab)
        endif
        idur = inumsamples / isr
        
        ispeed = icompensatesr==1 ? isr/sr : 1
        know init istart
        if inumouts == 0 then
            ; not a gen1 table, fail
            initerror sprintf("Table %d was not generated via gen1", isndtab)
        endif

        kidx init 0
        aenv = makePresetEnvelope(ifadein, ifadeout, ifadeshape, igain)

        if inumouts == 1 then
            a1 flooper2 1, ispeed*kspeed, istart, idur, ixfade, isndtab, istart
            a1 *= aenv
            ipos = ipos == -1 ? 0 : ipos
            aout1, aout2 pan2 a1, ipos
        elseif inumouts == 2 then
            a1, a2 flooper2 1, ispeed*kspeed, istart, idur, ixfade, isndtab, istart
            ipos = ipos < 0 ? 0.5 : ipos
            aout1, aout2 panstereo a1, a2, ipos
            aout1 *= aenv
            aout2 *= aenv
        else
            initerror sprintf("Multichannel samples (> 2, got %d) not supported yet", inumouts)
        endif
        outch ichan, aout1, ichan+1, aout2
        
        know += ionecycle * kspeed
        imaxtime = idur - ifade - ionecycle
        if iloop == 0 && know >= imaxtime then
            turnoff
        endif   
        """,
        envelope=False,
        routing=False,
        builtin=True),

    PresetDef(
        '_sing',
        description="Simple vowel singing simulation",
        init = r"""
        gi__formantFreqs__[] fillarray \
            668, 1191, 2428, 3321, 4600, \  ; A
            327, 2157, 2754, 3630, 4600, \  ; E 
            208, 2152, 3128, 3425, 4200, \  ; I
            335, 628, 2689, 3515, 4200, \   ; O
            254, 796, 2515, 3274, 4160      ; U
                       
        gi__formantDbs__[] fillarray   \
            28, 28, 22, 20, 20, \
            15, 25, 24, 20, 23, \
            10, 20, 27, 26, 20, \
            15, 18, 5,  7,  12, \
            12, 10, 6,  5,  12
                                   
        gi__formantBws__[] fillarray   \
            80, 90, 120, 130, 140, \
            60, 100, 120, 150, 200, \
            60, 90, 100, 120, 120, \
            40, 80, 100, 120, 120, \
            50, 60, 170, 180, 200
            
        gi__formantAmps__[] maparray gi__formantDbs__, "ampdb"
        reshapearray gi__formantFreqs__, 5, 5
        reshapearray gi__formantAmps__, 5, 5
        reshapearray gi__formantBws__, 5, 5
        """,
        audiogen = r"""
        |kx=0, ky=0, kvibamount=1|
        kvibfreq = linseg:k(0, 0.1, 0, 0.5, 4.5) * randomi:k(0.9, 1.1, 2)
        kvibsemi = linseg:k(0, 0.4, 0, 2.1, 0.25) * randomi:k(0.9, 1.1, 10)
        kvib = oscil:k(kvibsemi/2, kvibfreq) - kvibsemi/2
        kpitch = lag:k(kpitch, 0.2) + kvib*kvibamount
        asource = butterlp:a(vco2:a(kamp, mtof(kpitch)), 4000)
        kcoords[] fillarray 0, 0, 1,       \  ; A
                            0.5, 0.5, 0.3, \  ; E
                            1, 0, 1,       \  ; I
                            0, 1, 1,       \  ; O
                            1, 1, 1           ; U
        	
        kweights[] presetinterp kx, ky, kcoords, 0.2
        kformantFreqs[] weightedsum gi__formantFreqs__, kweights
        kformantBws[]   weightedsum gi__formantBws__, kweights
        kformantAmps[]  weightedsum gi__formantAmps__, kweights
        kformantFreqs poly 5, "lag", kformantFreqs, 0.2
        kformantAmps  poly 5, "lag", kformantAmps, 0.2
        aformants[] poly 5, "resonx", asource, kformantFreqs, kformantBws, 2, 1
        aformants *= kformantAmps
        aout1 = sumarray(aformants) * 0.1
        """,
        builtin=True)
]


soundfontGeneralMidiPresets = {
    'piano': (0, 0),     # "Yamaha Grand Piano"
    'clarinet': (0, 71),
    'oboe': (0, 68),
    'flute': (0, 73),
    'violin': (0, 40),
    'reedorgan': (0, 20)
}


_builtinSoundfonts = {
    # relative paths are relative to the package root
    '_piano': ('data/sf2/SalC5Light2.sf2', (0, 0), "Default piano sound")
}


def builtinSoundfonts() -> dict:
    """
    Get the paths of the builtin soundfonts

    Returns:
        a dict of {name: soundfontpath}, where *soundfontpath* is the absolute path
        to a sf2 file, ensuring that this file exists
    """
    from maelzel.dependencies import maelzelRootFolder
    root = maelzelRootFolder()
    out = {}
    for presetname, (relpath, preset, description) in _builtinSoundfonts.items():
        path = root / relpath
        if path.exists():
           out[presetname] = (path, preset, description)
    return out
