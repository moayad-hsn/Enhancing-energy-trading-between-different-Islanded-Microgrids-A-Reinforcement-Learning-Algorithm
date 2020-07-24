data=readtable("Day1.csv");

Tss=5e-6;

P=1.7e6;
U=380;
f=50;
fsw=5e3;

Cfmax=(0.05*P)/(2*pi*f*U^2);
Increment_MPPT= 0.01;     % Increment value used to increase/decrease Vdc_ref
Limits_MPPT= [ 583 357 ];

timeStamp = data.TimeStamp;
irrad = data.SolarSensor_solar_irradiance_Avg;
temp = data.Hygro_Thermo_temperature_Avg;
windSpeed = data.TopAnemometer_wind_speed_Avg;
R=1e-12;
L=2e-2;
C=550e-6;
Lf=(0.1*U^2)/(2*pi*f*P);
RLf=Lf*100;


