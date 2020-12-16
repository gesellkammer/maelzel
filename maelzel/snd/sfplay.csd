<CsoundSynthesizer>
<CsOptions>
-odac -+rtaudio=jack

</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 128
nchnls = 2
0dbfs  = 1

#define _(label) ##

#ifdef OSCPORT
	gioscport = $OSCPORT
#else
	gioscport = 45331
#endif

#ifdef SFPATH
	Spath = $SFPATH
#else
	Spath = "/home/em/flat/snd/sf2/grand-piano.sf2"
#endif

isfHandle sfload Spath
giPreset  sfpreset $_(progNum)0, $_(bank)0, isfHandle, $_(presetindex)0

giosc OSCinit gioscport 

alwayson "io", 0

instr 1	
	imidi   init p4
	ivel 	  init p5
	ipreset = giPreset
	iamp = 0.00015 
	iamp = iamp * ivel * 1/128
	kfreq mtof imidi
	a1, a2		sfplay3		ivel, imidi, iamp, kfreq, ipreset, 1
	aenv			linsegr		1, 1, 1, .1, 0
	
	a1 *= aenv
	a2 *= aenv
	outs a1, a2
endin

instr io
	kdur, kmidi, kvel init 0
	; /note dur midi vel (dur can be -1, which means until noteoff)
nextmsg:
	kk OSClisten giosc, "/note", "fff", kdur, kmidi, kvel
	if kk > 0 then
		event "i", 1 + int(kmidi * 100)/10000, 0, kdur, kmidi, kvel
		kgoto nextmsg
	endif
	
	kk OSClisten giosc, "/noteoff", "f", kmidi
	if kk > 0 then
		turnoff2 1 + int(kmidi *100)/10000, 4, 1
	endif
endin
</CsInstruments>
<CsScore>
f0  3600		

</CsScore>
</CsoundSynthesizer>
