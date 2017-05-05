board = {};
for address = 0x0400, 0x04c7 do 
	local v = memory.readbyte(address);
	if(v==0x00EF) then
		table.insert(board, 0);
	else 
		table.insert(board, 0.5);
	end
end
print(board);
