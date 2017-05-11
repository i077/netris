require("makemoves")
function deepCopy(object)
    local lookup_table = {}
    local function _copy(object)
        if type(object) ~= "table" then
            return object
        elseif lookup_table[object] then
            return lookup_table[object]
        end
        local new_table = {}
        lookup_table[object] = new_table
        for index, value in pairs(object) do
            new_table[_copy(index)] = _copy(value)
        end
        return setmetatable(new_table, getmetatable(object))
    end
    return _copy(object)
end

function readmaneuvers()--creates maneuver list
    local file = assert(io.open("testmaneuvers.txt",'r'));
    --file saved in current directory, each line is one maneuver, moves not delineated
    io.input(file);
    local maneuverlist = {};
    for line in io.lines() do
        local moves = {};
        --print(string.len(line));
        for i = 1,string.len(line) do
            table.insert(moves,string.sub(line,i,i));
            --print(moves[i]);
        end
        table.insert(maneuverlist,moves);
    end
    return maneuverlist;
end
--END OF HELPER FUNCTIONS

----testmovelist = {'a','l','a','r','b','l',};
----inputs = makeMoves(testmovelist);
maneuvers = {};
--testmaneuverlist = {{'a','l','a','l'},{'b','r','b'},{'l','l','l','b','r'},{'l','r','l','r','l','r','l','r','r','r','b','a','l'},{'u','u','d','d','l','r','l','r','b','a'}};
testmaneuverlist = readmaneuvers();
for a = 1,#testmaneuverlist do
	table.insert(maneuvers, makeMoves(testmaneuverlist[a]));
end
movenum=1;
maneuvernum=0;

while(true) do 

    piecenum = memory.readbyte(0x001A); --tracks "number of pieces this poweron" value at address 0x001A in memory
    if(piecenum ~= lastpiecenum) then --when a new piece falls, restart the move queue
        movenum=1;
		maneuvernum=maneuvernum+1;
        lastpiecenum=piecenum; --so it won't do anything again until a new piece falls
    end

    --input control 
	currentmaneuver=maneuvers[maneuvernum];
	if(currentmaneuver) then
		currentmove=currentmaneuver[movenum];
		joypad.set(1,{A=nil, up=nil, left=nil, B=nil, select=nil, right=nil, down=nil, start=nil});
		if(currentmove) then
			joypad.set(1,currentmove);
		end
	end
	movenum=movenum+1;

	FCEU.frameadvance();
end
