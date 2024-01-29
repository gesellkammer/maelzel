from .presetdef import PresetDef

builtinPresets = [
    PresetDef(
        'simplesin',
        'aout1 oscili a(kamp), mtof(lag(kpitch, 0.01))',
        description="simplest sine wave",
        builtin=True
    ),

    PresetDef(
        'sin',
        "aout1 oscili a(kamp), mtof(lag(kpitch+ktransp, klag))",
        args=dict(ktransp=0, klag=0.1),
        description="transposable sine wave",
        builtin=True
    ),

    PresetDef(
        'tri', r'''
        |ktransp=0, klag=0.1, kcutoffratio=0, kfilterq=3|
        ; transposable triangle wave with optional lowpass-filter
        ; Args:
        ;   ktransp: transposition interval
        ;   klag: lag time when modifying pitch
        ;   kcutoffratio: cutoff frequency of the filter as a factor of the osc frequency
        ;   kfilterq: filter resonance
        kfreq = mtof:k(lag(kpitch + ktransp, klag))
        aout1 = vco2(1, kfreq,  12) * a(kamp)
        if kcutoffratio > 0 then
           aout1 = K35_lpf(aout1, kfreq*kcutoffratio, kfilterq)
        endif
        ''',
        description="transposable triangle wave with optional lowpass-filter",
        builtin=True
    ),

    PresetDef(
        'saw', r'''
        |ktransp=0, klag=0.1, kcutoffratio=0, kfilterq=3|
        ; Transposable saw with optional low-pass filtering
        ;  Args:
        ;    ktransp: transposition interval
        ;    klag: lag time when modifying pitch
        ;    kcutoffratio: filter cutoff frequency as factor of the osc frequency
        ;    kfilterq: filter resonance
        kfreq = mtof:k(lag(kpitch + ktransp, klag))
        asig = vco2(1, kfreq, 0) * a(kamp)
        aout1 = kcutoffratio == 0 ? asig : K35_lpf(asig, kfreq*kcutoffratio, kfilterq)
        ''',
        description="Transposable saw with optional low-pass filtering",
        builtin=True
    ),

    PresetDef(
        'sqr', r'''
        |ktransp=0, klag=0.1, kcutoff=0, kresonance=0.2|
        ; square wave with optional filtering
        ; Args:
        ;    kcutoff: filter cutoff frequency
        ;    kresonance: resonance of the filter
        aout1 = vco2(1, mtof(lag(kpitch+ktransp, klag), 10) * a(kamp)
        aout1 = kcutoff == 0 ? aout1 : moogladder aout1, lag(kcutoff, 0.1), kresonance
        ''',
        description="square wave with optional filtering",
        builtin=True
    ),

    PresetDef(
        'pulse', r"""
        |ktransp=0, klag=0.1, kpwm=0.5|
        ; transposable pulse with pwm
        ; Args:
        ;   ktransp: transposition
        ;   klag: lag time for pitch
        ;   kpwm: pwm between 0-1
        aout1 vco2 kamp, mtof:k(lag:k(kpitch+ktransp, klag), 2, kpwm
        """,
        builtin=True
    ),

    PresetDef(
        '.click', r"""
        |itransp=24, idecay=0.1|
        ; Default preset used when rendering a click-track
        ; Args:
        ;   itransp: transposition interval
        aclickenv expseg db(-120), 0.01, 1, idecay, db(-120)
        aout1 = oscili:a(aclickenv, mtof:k(kpitch+itransp))
        """,
        description="Default preset used when rendering a click-track",
        builtin=True
    ),

    PresetDef(
        '_clip_diskin', code=r'''
        |ipath, isndfilechan=-1, kspeed=1, iskip=0, iwrap=0, iwinsize=4|
        ; Builtin-in preset to play a clip using diskin
        ; Args:
        ;   ipath: the path to the soundfile, as set via strSet
        ;   isndfilechan: if given, play only this channel from the soundfile
        ;   kspeed: the speed of the playback
        ;   iskip: number of seconds to skip from beginning (assuming kspeed=1)
        ;   iwrap: if non-zero, locations wrap (results in looping)
        ;   iwinsize: interpolation size (1=no interpol, 2=linear, 4=cubic, 8=sinc)
        Spath = strget(ipath)
        if strlen(Spath) == 0 then
            initerror sprintf("Soundfile '%s' not found", Spath)
        endif
        
        iformat = 0
        ibufsize = 0   ; 0 sets the buffer to the default of 4096
        inumchannels = filenchnls(Spath)
        if inumchannels == 0 || inumchannels > 2 then
            initerror sprintf("Multichannel samples (> 2, got %d) not supported yet", inumchannels)
        endif
        
        aenv = makePresetEnvelope(ifadein, ifadeout, ifadekind)
        aenv *= kgain
        asig[] diskin2 Spath, kspeed, iskip, iwrap, iformat, iwinsize, ibufsize
        if isndfilechan >= 0 then
            a1 = asig[isndfilechan]
            kpos = kpos == -1 ? 0 : kpos
            a1 *= aenv
            aout1, aout2 pan2 a1, kpos
        elseif inumchannels == 1 then
            a1 = asig[0]
            a1 *= aenv
            kpos = kpos == -1 ? 0 : kpos
            aout1, aout2 pan2 a1, kpos
        elseif inumchannels == 2 then
            asig *= aenv
            aout1, aout2 panstereo asig[0], asig[1], kpos
        endif
        outch ichan, aout1, ichan+1, aout2
        ''',
        routing=False,
        envelope=False,
        aliases={'speed': 'kspeed'}
    ),

    PresetDef(
        '_playtable', code=r"""
        |isndtab=0, istart=0, kspeed=1, ixfade=-1|
        ; Built-in presetdef to playback a table
        ; Args:
        ;   isndtab: table number to play
        ;   istart: skip time
        ;   kspeed: playback speed
        ;   ixfade: crossfade time, if negative no looping
        iloop = ixfade >= 0 ? 1 : 0
        inumouts = ftchnls(isndtab)
        inumsamples = nsamp(isndtab)
        isr = ftsr(isndtab)
        ionecycle = ksmps/sr
        
        if isr <= 0 then
            initerror sprintf("Could not determine sr of table %d", isndtab)
        endif
        idur = inumsamples / isr
        
        know init istart
        if inumouts == 0 then
            ; not a gen1 table, fail
            initerror sprintf("Table %d was not generated via gen1", isndtab)
        endif

        kidx init 0
        aenv = makePresetEnvelope(ifadein, ifadeout, ifadekind)
        aenv *= kgain

        if inumouts == 1 then
            a1 flooper2 1, kspeed, istart, idur, ixfade, isndtab, istart
            a1 *= aenv
            ipos = ipos == -1 ? 0 : ipos
            aout1, aout2 pan2 a1, ipos
        elseif inumouts == 2 then
            a1, a2 flooper2 1, kspeed, istart, idur, ixfade, isndtab, istart
            ipos = ipos < 0 ? 0.5 : ipos
            aout1, aout2 panstereo a1, a2, ipos
            aout1 *= aenv
            aout2 *= aenv
        else
            initerror sprintf("Multichannel samples (> 2, got %d) not supported yet", inumouts)
        endif
        outch ichan, aout1, ichan+1, aout2
        
        know += ionecycle * kspeed
        imaxtime = idur - ifadeout - ionecycle
        if iloop == 0 && know >= imaxtime then
            turnoff
        endif   
        """,
        envelope=False,
        routing=False,
        builtin=True
    ),

    PresetDef(
        '.sing', description="Simple vowel singing simulation",
        init=r"""
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
        code=r"""
        |kx=0, ky=0, kvibrange=0.25, kvibfreq=4.5, ivibstart=0.5, ipitchlag=0.2|
        ; Simple vowel singing simulation
        ; Args:
        ;   kx: x coordinate, from 0 to 1, A=(0;0), E=(0.5;0.5), I=(1;0), O=(0;1), U=(1;1)
        ;   ky: y coordinate, from 0 to 1, A=(0;0), E=(0.5;0.5), I=(1;0), O=(0;1), U=(1;1)
        ;   kvibrange: vibrato range in semitones
        ;   ivibstart: start time of vibrato
        ;   ivibfreq: vibrato frequency
        ;   ipitchlag: time lag for pitch modifications
        
        knoVib = lag:k(trighold(changed2(kpitch), ivibstart*0.8), ivibstart*0.2)
        kvibfreq2 = linseg:k(0, ivibstart*0.25, 0, ivibstart*0.75, 1) * randomi:k(0.9, 1.1, 2) * (1 - knoVib) * kvibfreq 
        kvibamount = linseg:k(0, ivibstart*0.2, 0, ivibstart*0.8, 1) * randomi:k(0.9, 1.1, 10)
        kvibsemi = kvibamount * kvibrange
        kvib = oscil:k(kvibsemi/2, kvibfreq2) - kvibsemi/2
        kpitch2 = lag:k(kpitch, ipitchlag) + kvib
        asource = butterlp:a(vco2:a(kamp, mtof(kpitch2)), 5000)
        kcoords[] fillarray 0, 0, 1,       \  ; A
                            0.5, 0.5, 0.3, \  ; E
                            1, 0, 1,       \  ; I
                            0, 1, 1,       \  ; O
                            1, 1, 1           ; U

        kweights[] presetinterp kx, ky, kcoords, 0.2
        kformantFreqs[] weightedsum gi__formantFreqs__, kweights
        kformantBws[]   weightedsum gi__formantBws__, kweights
        kformantAmps[]  weightedsum gi__formantAmps__, kweights
        kformantFreqs poly 5, "lag", kformantFreqs, ipitchlag
        kformantAmps  poly 5, "lag", kformantAmps, ipitchlag
        aformants[] poly 5, "resonx", asource, kformantFreqs, kformantBws, 2, 1
        aformants *= kformantAmps
        aout1 = sumarray(aformants) * 0.1
        """,
        builtin=True
    )
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
    '.piano': ('sf2/SalC5Light2.sf2', (0, 0), "Default piano sound")
}


def builtinSoundfonts() -> dict:
    """
    Get the paths of the builtin soundfonts

    Returns:
        a dict of {name: soundfontpath}, where *soundfontpath* is the absolute path
        to a sf2 file, ensuring that this file exists
    """
    # from maelzel.dependencies import maelzelRootFolder
    from maelzel.dependencies import dataPath
    datadir = dataPath()
    out = {}
    for presetname, (relpath, preset, description) in _builtinSoundfonts.items():
        path = datadir / relpath
        if path.exists():
            out[presetname] = (path, preset, description)
    return out
