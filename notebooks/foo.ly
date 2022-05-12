\version "2.22.1"
% automatically converted by musicxml2ly from /tmp/music21/tmpel6caub8.musicxml
\pointAndClickOff

\header {
    encodingsoftware =  "music21 v.7.1.0"
    encodingdate =  "2022-03-31"
    }

#(set-global-staff-size 20.0)
\paper {
    
    }
\layout {
    \context { \Score
        autoBeaming = ##f
        }
    }
PartPcefadfFiveThreeTwoFivedThreeTwoSevenNineeNineEightTwoEightaSevenaSevenOnecfEightfSixThreeEightVoiceOne = 
\relative c' {
    \clef "treble" \numericTimeSignature\time 4/4 | % 1
    \tempo 4=80 c1 | % 2
    \time 5/4  c1 ~ ^ "hola!" c4 | % 3
    \time 3/8  \tempo 4=120 c4. | % 4
    \time 4/8  c2 | % 5
    \time 3/4  c2. \bar "|."
    }


% The score definition
\score {
    <<
        
        \new Staff
        <<
            
            \context Staff << 
                \mergeDifferentlyDottedOn\mergeDifferentlyHeadedOn
                \context Voice = "PartPcefadfFiveThreeTwoFivedThreeTwoSevenNineeNineEightTwoEightaSevenaSevenOnecfEightfSixThreeEightVoiceOne" {  \PartPcefadfFiveThreeTwoFivedThreeTwoSevenNineeNineEightTwoEightaSevenaSevenOnecfEightfSixThreeEightVoiceOne }
                >>
            >>
        
        >>
    \layout {}
    % To create MIDI output, uncomment the following line:
    %  \midi {\tempo 4 = 80 }
    }

