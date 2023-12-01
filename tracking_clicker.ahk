length := 577
interval := 600
quickOffset := 20
active := 0
showed := 0
init := 0
granularity := 20

^+p::{
    global active
    num := 0
    done := False
    while num < length and not done{
        active := 1
        if not GetKeyState("Space") and not GetKeyState("Shift"){
            TrackActive
            num++
            Click
        } else {
            TrackPause
        }
        Loop granularity {
            if GetKeyState("Escape"){
                done := True
                Break
            }   
            Sleep(interval/granularity)
        }
    }
    TrackInactive
    active := 0
}


^+l::{
    global length
    l := ""
    complete := False
    while !complete{
        l := InputBox("Input Track Length", "Please enter a positive integer:",,length).Value
        if (not IsInteger(l)){
            MsgBox("Track Length must be an integer!")
        } else if (l <= 0) {
            MsgBox("Track Length must be a positive nonzero integer!")
        } else {
            complete := True
        }
    }
    length := l + 0
}

^+t::{
    global interval
    i := ""
    complete := False
    while !complete{
        i := InputBox("Input Click Interval", "Please enter a positive integer (milliseconds):",,interval).Value
        if (not IsInteger(i)){
            MsgBox("Click Interval must be an integer!")
        } else if (i < 0) {
            MsgBox("Click Interval must be a positive integer!")
        } else {
            complete := True
        }
    }
    interval := i + 0
}

#HotIf active == 1
*=:: global interval -= quickOffset
*-:: global interval += quickOffset
#HotIf

^+o::{
if showed {
    HideWindow
} else {
    ShowWindow
}
}


ShowWindow(){
    if (!init){
        MakeWindow
        TrackInactive
    } else {
        ShowWindow
    }
}

HideWindow(){
    showed := 0
}


MakeWindow(){
    global StatusWindow
    global MyText
    global init
    global showed
    ; WinGetTitle(winid, A)
    init := 1
    showed := 1
    DetectHiddenWindows True
    StatusWindow := Gui("-Caption +AlwaysOnTop +LastFound")
    StatusWindow.SetFont("s40")
    MyText := StatusWindow.AddText("x-3 y-18",Chr(9679))
    MyText.OnEvent("Click",Move)
    ;gMove tells it to call "move" from click_test.ahk. ENSURE THE WINDOW IS STILL MOVEABLE!
    WinSetTransColor(0xF0F0F0)
    yloc := A_ScreenHeight-99
    xloc := A_ScreenWidth-38
    StatusWindow.Show(Format("w28 h28 x{} y{}",xloc,yloc))
}

TrackActive(){
    if init{
        MyText.SetFont("cGreen")
    }
}

TrackInactive(){
    if init{
        MyText.SetFont("cRed")
    }
}

TrackPause(){
    if init{
        MyText.SetFont("cYellow")
    }
}

Move(Ctrl,Info){
	PostMessage(0xA1, 2)
}