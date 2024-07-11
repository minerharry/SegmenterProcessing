F5::
if !init
{
	WinGetTitle, winid, B
	init := 1
	DetectHiddenWindows, On
	Gui Font, s40
	Gui,add,Text,vMyText x-3 y-18 gMove,% Chr(9679)
	Gui -Caption +AlwaysOnTop +LastFound
	WinSet,TransColor,0xF0F0F0
    yloc := A_ScreenHeight-99
    xloc := A_ScreenWidth-38
	Gui Show,w28 h28 x%xloc% y%yloc%
}
if (t:=!t){
	Gui Font, cGreen
	GuiControl, Font, MyText
	; SetTimer, PressTheKey, 10000
} else {
	Gui Font, cRed
	GuiControl, Font, MyText
	; SetTimer, PressTheKey, Off
}
return

PressTheKey:
	ControlSend,, g, %winid%
return

Move:
	PostMessage, 0xA1, 2
Return