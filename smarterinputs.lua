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

function makeMoves(movelist) --takes a list of chars

	--[[
		
		movelist format:
		each button (left column) is represented by a char (right column)

		A = a
		B = b
		UP = u
		DOWN = d
		LEFT = l
		RIGHT = r
		Start = t
		Select = s

	--]]
	
	blanktable = {A=false, up=false, left=false, B=false, select=false, right=false, down=false, start=false}; --An empty inputtable, tells the emulator to do nothing
	finalinputs = {};--a list 

	for i, move in ipairs(movelist) do
		inputtable=deepCopy(blanktable);
		if move == 'a' then
			inputtable.A=true;
		elseif move == 'b' then
			inputtable.B=true;
		elseif move == 'u' then
			inputtable.up=true;
		elseif move == 'd' then
			inputtable.down=true;
		elseif move == 'l' then
			inputtable.left=true;
		elseif move == 'r' then
			inputtable.right=true;
		elseif move == 't' then
			inputtable.start=true;
		elseif move == 's' then
			inputtable.select=true;
		else
			
		end
		
		table.insert(finalinputs, inputtable);
		for a = 1,16 do
			table.insert(finalinputs, deepCopy(blanktable));
		end

	end

	return finalinputs;

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
