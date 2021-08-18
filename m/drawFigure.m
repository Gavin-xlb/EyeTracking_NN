ec_x = linspace(-4, 4, 60);ec_y = linspace(-6, 6, 60);
[mesh_x, mesh_y]=meshgrid(ec_x,ec_y);
z=-17.65* (mesh_x .* mesh_x)-1.255*(mesh_x .* mesh_y)+10.66*( mesh_y .* mesh_y )-319.8*mesh_x+77.53*mesh_y+486.0;

figure(1)
surf(mesh_x,mesh_y,z)
xlabel('x');ylabel('y');zlabel('z');
figure(2)
z=-16.53* (mesh_x .* mesh_x)+35.25*(mesh_x .* mesh_y)-51.20*( mesh_y .* mesh_y )-81.88*mesh_x+43.57*mesh_y+711.6;
surf(mesh_x,mesh_y,z)
xlabel('x');ylabel('y');zlabel('z');

%{
line_x = linspace(0, 0, 60);
z = -37.37 * ec_y .* ec_y + 214 * ec_y + 921.6;
plot3(line_x, ec_y, z);
line_y = linspace(0, 0, 60);
z = -2.328 * ec_x .* ec_x -131 * ec_x + 921.6;
plot3(ec_x, line_y, z');
xlabel('x');ylabel('y');zlabel('z');
%}
