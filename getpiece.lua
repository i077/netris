function getPiece(n) -- returns table with information about this piece
	-- every piece consists of 4 blocks
	-- so piece will contain the information about these blocks
	-- for every block is the offset compared to the x,y position of the piece
	-- the x,y is written in a 4x4 plane of blocks, 1x1 being the topleft block
	-- x,y is the position as given by tetris
	local piece = {};
	if (n == 0) then -- T right
		piece.pos = {{2,1},{1,2},{2,2},{3,2}};
		piece.rel = {{0,-1},{-1,0},{0,0},{1,0}};
		piece.anchor = {2,2};
	elseif (n == 1) then -- T up
		piece.pos = {{1,1},{1,2},{2,2},{1,3}};
		piece.rel = {{0,-1},{0,0},{1,0},{0,1}};
		piece.anchor = {1,2};
	elseif (n == 2) then -- T down
		piece.pos = {{1,1},{2,1},{3,1},{2,2}};
		piece.rel = {{-1,0},{0,0},{1,0},{0,1}};
		piece.anchor = {2,1};
	elseif (n == 3) then -- T left
		piece.pos = {{2,1},{1,2},{2,2},{2,3}};
		piece.rel = {{0,-1},{-1,0},{0,0},{0,1}};
		piece.anchor = {2,2};
	elseif (n == 4) then -- J left
		piece.pos = {{2,1},{2,2},{1,3},{2,3}};
		piece.rel = {{0,-1},{0,0},{-1,1},{0,1}};
		piece.anchor = {2,2};
	elseif (n == 5) then -- J up
		piece.pos = {{1,1},{1,2},{2,2},{3,2}};
		piece.rel = {{-1,-1},{-1,0},{0,0},{1,0}};
		piece.anchor = {2,2};
	elseif (n == 6) then -- J right
		piece.pos = {{1,1},{2,1},{1,2},{1,3}};
		piece.rel = {{0,-1},{1,-1},{0,0},{0,1}};
		piece.anchor = {1,2};
	elseif (n == 7) then -- J down
		piece.pos = {{1,1},{2,1},{3,1},{3,2}};
		piece.rel = {{-1,0},{0,0},{1,0},{1,1}};
		piece.anchor = {2,1};
	elseif (n == 8) then -- Z horz
		piece.pos = {{1,1},{2,1},{2,2},{3,2}};
		piece.rel = {{-1,0},{0,0},{0,1},{1,1}};
		piece.anchor = {2,1};
	elseif (n == 9) then -- Z vert
		piece.pos = {{2,1},{1,2},{2,2},{1,3}};
		piece.rel = {{1,-1},{0,0},{1,0},{0,1}};
		piece.anchor = {1,2};
	elseif (n == 10) then -- O
		piece.pos = {{1,1},{2,1},{1,2},{2,2}};
		piece.rel = {{-1,0},{0,0},{-1,1},{0,1}};
		piece.anchor = {2,1};
	elseif (n == 11) then -- S horz
		piece.pos = {{2,1},{3,1},{1,2},{2,2}};
		piece.rel = {{0,0},{1,0},{-1,1},{0,1}};
		piece.anchor = {2,1};
	elseif (n == 12) then -- S vert
		piece.pos = {{1,1},{1,2},{2,2},{2,3}};
		piece.rel = {{0,-1},{0,0},{1,0},{1,1}};
		piece.anchor = {1,2};
	elseif (n == 13) then -- L right
		piece.pos = {{1,1},{1,2},{1,3},{2,3}};
		piece.rel = {{0,-1},{0,0},{0,1},{1,1}};
		piece.anchor = {1,2};
	elseif (n == 14) then -- L down
		piece.pos = {{1,1},{2,1},{3,1},{1,2}};
		piece.rel = {{-1,0},{0,0},{1,0},{-1,1}};
		piece.anchor = {2,1};
	elseif (n == 15) then -- L left
		piece.pos = {{1,1},{2,1},{2,2},{2,3}};
		piece.rel = {{-1,-1},{0,-1},{0,0},{0,1}};
		piece.anchor = {2,2};
	elseif (n == 16) then -- L up
		piece.pos = {{3,1},{1,2},{2,2},{3,2}};
		piece.rel = {{1,-1},{-1,0},{0,0},{1,0}};
		piece.anchor = {2,2};
	elseif (n == 17) then -- I vert
		piece.pos = {{1,1},{1,2},{1,3},{1,4}};
		piece.rel = {{0,-2},{0,-1},{0,0},{0,1}};
		piece.anchor = {1,3};
	elseif (n == 18) then -- I horz
		piece.pos = {{1,1},{2,1},{3,1},{4,1}};
		piece.rel = {{-2,0},{-1,0},{0,0},{1,0}};
		piece.anchor = {3,1};
	else
		return nil;
	end;
	piece.id = n;
	return piece;
end;


