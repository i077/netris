function readboard()
	board = {};
	for address = 0x0400, 0x04c7 do 
		local v = memory.readbyte(address);
		if(v==0x00EF) then
			table.insert(board, 0);
		else 
			table.insert(board, 0.5);
		end
	end

	local current = getPiece(memory.readbyte(0x0062)); 
	local x = memory.readbyte(0x0040)+1;
	local y = memory.readbyte(0x0041)+1;
	local width = 10;
	local height = 20;
	if (current) then
		for i = 1,4 do
			tempx = x + current.rel[i][1];
			tempy = y + current.rel[i][2];
			index = tempx + (tempy - 1)*width;
			board[index]=1;
		end
	end
	
	return board;
end
