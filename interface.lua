require("readboard")
require("getpiece")
require("makemoves")

while(true) do
    
    readboard()--prints the board status for the NN to read

    --Read the file
    f = nil
    flag = 0
    while (flag == 0) do
        while (f == nil) do
            f = assert(io.open("moves.txt", "r+"))
        end
        io.input(f)
        flag = io.read("*n")
        if(flag==0) then
            io.close()
        end
    end
    
    move = string.sub(io.read(), 1, 1)
    nextmove = makeMoves({move})--make the inputtable
    joypad.write(1, nextmove)

    
    io.close(f)
    FCEU.frameadvance()
end
