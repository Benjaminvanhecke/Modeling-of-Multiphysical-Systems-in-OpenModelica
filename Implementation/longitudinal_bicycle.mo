model Minimalistic_1
  import Modelica.Units.SI;
  import Modelica.Constants.g_n;
  parameter SI.Mass m = 80 "Mass of vehicle";
  parameter SI.Length R = 0.35 "Radius of wheel";
  parameter SI.Area A =0.67 "0.59 Cross section of vehicle";
  parameter Real Cd = 0.67 "0.59 Drag resistance coefficient";
  parameter SI.Density rho = 1.225 "Density of air";
  parameter SI.Velocity vWind = 0 "Constant wind velocity";
  //Check nominal force
  parameter Real CrConstant = 0.009 "Constant Cr";
  parameter Real inclinationConstant = 0 "Constant inclination = tan(angle)";
  parameter SI.Velocity vReg = 1e-3 "Velocity for regularization around 0";
  Modelica.Blocks.Sources.CombiTimeTable Table(tableOnFile = true, tableName = "TimeTorqueInclinationSpeed_rad_tanPrediction", fileName = "C:/Users/skipp/Documents/school/ModelicaProject/Implementation/TimeTorqueInclinationSpeed_rad_tanPrediction.txt", extrapolation = Modelica.Blocks.Types.Extrapolation.HoldLastPoint, columns = {2, 3, 4, 5}) annotation(
    Placement(transformation(origin = {-86, 12}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Mechanics.Translational.Sources.QuadraticSpeedDependentForce fDrag(final ForceDirection = false, final f_nominal = -Cd*A*rho*Vref^2/2, final useSupport = true, final v_nominal = Vref) "Drag resistance" annotation(
    Placement(transformation(origin = {14, -31}, extent = {{50, -40}, {70, -20}})));
  Modelica.Mechanics.Translational.Components.RollingResistance fRoll(final CrConstant = CrConstant, final fWeight = m*g_n, final inclinationConstant = inclinationConstant, final reg = Modelica.Blocks.Types.Regularization.Linear, final useCrInput = true, final useInclinationInput = true, final v0 = vReg) "Rolling resistance" annotation(
    Placement(transformation(origin = {22, -21}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Mechanics.Translational.Sources.Force fGrav "Inclination resistance" annotation(
    Placement(transformation(origin = {74, 17}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Mechanics.Rotational.Components.Inertia inertia(final J = 0) annotation(
    Placement(transformation(origin = {16, 3}, extent = {{-50, 50}, {-30, 70}})));
  Modelica.Mechanics.Translational.Components.IdealRollingWheel idealRollingWheel(final radius = R) annotation(
    Placement(transformation(origin = {4, 3}, extent = {{-10, 50}, {10, 70}})));
  Modelica.Mechanics.Translational.Components.Mass mass(final m = m, v(start = 9.34)) annotation(
    Placement(transformation(origin = {34, 3}, extent = {{30, 50}, {50, 70}})));
  Modelica.Blocks.Math.Gain gravForceGain(final k = -m*g_n) annotation(
    Placement(transformation(origin = {44, 17}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Math.Sin sin annotation(
    Placement(transformation(origin = {12, 17}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Sources.Constant Cr(k = CrConstant) annotation(
    Placement(transformation(origin = {-84, -28}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Mechanics.Rotational.Sources.Torque torque annotation(
    Placement(transformation(origin = {-59, 63}, extent = {{-5, -5}, {5, 5}})));
  Modelica.Blocks.Sources.Constant constWindSpeed1(k = vWind) annotation(
    Placement(transformation(origin = {-84, -70}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Mechanics.Translational.Sources.Speed speed annotation(
    Placement(transformation(origin = {16, -70}, extent = {{-10, -10}, {10, 10}})));
protected
  constant SI.Velocity Vref = 1;
public
  Modelica.Blocks.Math.Atan atan annotation(
    Placement(transformation(origin = {-24, 15}, extent = {{-10, -10}, {10, 10}})));
equation
  connect(idealRollingWheel.flangeT, mass.flange_a) annotation(
    Line(points = {{14, 63}, {64, 63}}, color = {0, 127, 0}));
  connect(gravForceGain.u, sin.y) annotation(
    Line(points = {{32, 17}, {23, 17}}, color = {0, 0, 127}));
  connect(gravForceGain.y, fGrav.f) annotation(
    Line(points = {{55, 17}, {62, 17}}, color = {0, 0, 127}));
  connect(inertia.flange_b, idealRollingWheel.flangeR) annotation(
    Line(points = {{-14, 63}, {-6, 63}}));
  connect(fRoll.cr, Cr.y) annotation(
    Line(points = {{10, -27}, {-34.5, -27}, {-34.5, -28}, {-73, -28}}, color = {0, 0, 127}));
  connect(fRoll.inclination, Table.y[2]) annotation(
    Line(points = {{10, -15}, {-74, -15}, {-74, 12}, {-75, 12}}, color = {0, 0, 127}));
  connect(torque.flange, inertia.flange_a) annotation(
    Line(points = {{-54, 63}, {-34, 63}}));
  connect(speed.flange, fDrag.support) annotation(
    Line(points = {{26, -70}, {48, -70}, {48, -71}, {74, -71}}, color = {0, 127, 0}));
  connect(constWindSpeed1.y, speed.v_ref) annotation(
    Line(points = {{-72, -70}, {4, -70}}, color = {0, 0, 127}));
  connect(mass.flange_b, fGrav.flange) annotation(
    Line(points = {{84, 63}, {84, 17}}, color = {0, 127, 0}));
  connect(fRoll.flange, fGrav.flange) annotation(
    Line(points = {{32, -21}, {32, -19.75}, {58, -19.75}, {58, -20}, {84, -20}, {84, 17}}, color = {0, 127, 0}));
  connect(fDrag.flange, fGrav.flange) annotation(
    Line(points = {{84, -61}, {84, 17}}, color = {0, 127, 0}));
  connect(atan.y, sin.u) annotation(
    Line(points = {{-12, 16}, {0, 16}, {0, 17}}, color = {0, 0, 127}));
  connect(atan.u, Table.y[2]) annotation(
    Line(points = {{-36, 16}, {-59.5, 16}, {-59.5, 12}, {-75, 12}}, color = {0, 0, 127}));
  connect(torque.tau, Table.y[1]) annotation(
    Line(points = {{-65, 63}, {-72, 63}, {-72, 16}, {-70.5, 16}, {-70.5, 12}, {-75, 12}}, color = {0, 0, 127}));
  annotation(
    Dialog(enable = false));
  annotation(
    uses(Modelica(version = "4.0.0")));
end Minimalistic_1;
