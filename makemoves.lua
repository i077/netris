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
