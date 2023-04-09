length := 577
interval := 600
quickOffset := 20
active := 0

^+p::
num := 0
done := False
while num < length and not done{
    active := 1
    if not GetKeyState("Space") and not GetKeyState("Shift"){
        num++
        Click
    }
    Loop 5 {
        if GetKeyState("Escape"){
            done := True
            Break
        }
        Sleep, interval/5
    }
}
active := 0
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

#If active == 1
*=::
interval -= quickOffset
Return

*-::
interval += quickOffset
Return