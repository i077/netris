require("getpiece");
require("readboard");

local width = 10;
local height = 20;

while(true) do
	board = readboard();
	for j=0,height-1 do
		for i=0,width-1 do
			if(board[j*width+i+1]==0.5) then
				gui.pixel(i,j+8,"red");
			elseif(board[j*width+i+1]==1) then
				gui.pixel(i,j+8,"blue");
			else
				gui.pixel(i,j+8,"white");
			end
		end
	end
	gui.drawbox(0,0,10,10,"green");
	FCEU.frameadvance();
end
