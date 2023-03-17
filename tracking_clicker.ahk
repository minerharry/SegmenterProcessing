length := 577
interval := 600

^+p::
num := 0
done := False
sleeptime := interval/5
while num < length and not done{
    if not GetKeyState("Space") and not GetKeyState("Shift"){
        num++
        Click
    }
    Loop 5 {
        if GetKeyState("Escape"){
            done := True
            Break
        }
        Sleep, sleeptime
    }
}
Return


^+l::
l := ""
complete := False
while !complete{
    InputBox, l, Input Track Length, Please enter a positive integer:,,,,,,,, %length%
    if (l is not integer){
        MsgBox, "Track Length must be an integer!"
    } else if (l <= 0) {
        MsgBox, "Track Length must be a positive nonzero integer!"
    } else {
        complete := True
    }
}
length := l + 0
Return

^+t::
i := ""
complete := False
while !complete{
    InputBox, i, Input Click Interval, Please enter a positive integer (milliseconds):,,,,,,,, %interval%
    if (i is not Integer){
        MsgBox, "Click Interval must be an integer!"
    } else if (i < 0) {
        MsgBox, "Click Interval must be a positive integer!"
    } else {
        complete := True
    }
}
interval := i + 0
Return