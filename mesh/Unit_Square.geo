// Gmsh project created on Mon Dec  3 23:10:47 2018
//+
Point(1) = {0.0, 0.0, 0, 1.0};
//+
Point(2) = {1.0, 0.0, 0, 1.0};
//+
Point(3) = {0.0, 1.0, 0, 1.0};
//+
Point(4) = {1.0, 1.0, 0, 1.0};
//+
Line(1) = {3, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 4};
//+
Line(4) = {4, 3};
//+
Line Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Characteristic Length {1, 2, 4, 3} = 0.09;
