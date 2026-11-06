	// ---- AssistZ lab: ground-truth pose log -------------------------------------------------------------------
	// Injected into the offline mission's CustomMission by Lab/make_lab_missions.ps1. The mission runs in a DayZDiag
	// SERVER (-server) with a DayZDiag client connected to it, so this is server-side code: there is no local
	// player and no camera here. Every 100 ms it logs the first connected player's position, body orientation and
	// the aim angles the server simulates for that player (HumanCommandWeapons base aiming = where the head/weapon
	// points relative to the body), so every captured frame and sound has a known answer.
	// Columns: game_ms = GetGame().GetTime() (ms since mission start), utc = wall clock (s resolution; fine alignment
	// against the capture uses a sync event, not this column).
	// Rows are buffered and APPENDED once a second (open, write, close) so the CSV is complete on disk while the game
	// runs; a handle held open is only flushed at close. (FileHandle is not an int in Enforce: assigning or comparing 0
	// fails to compile, so the "is logging" state is its own flag.)
	protected string m_AzPosePath;
	protected bool m_AzPoseOpen;
	protected float m_AzPoseAccum;
	protected float m_AzPoseFlushAccum;
	protected ref array<string> m_AzPoseRows = new array<string>;
	protected ref array<Man> m_AzPlayers = new array<Man>;
	protected bool m_AzMode6Seen;

	// ---- lab mode, chosen when make_lab_missions.ps1 builds the mission (-Mode): 0 = pose only, 1 = sound beacon,
	// 2 = scripted view turns, 3 = two WORLD-FIXED repeating beacons (front/back resolution while the view turns),
	// 4 = lab sniper: a server-spawned SVD fires near-misses at the player from a range ladder (crack-to-report delay),
	// 5 = REAL shooter: a second client ("Shooter", driven by the lab mod) fires the same ladder; the listener is "AssistZLab"
	//     (fails: the server rejects a second client on the same Steam ID),
	// 6 = DayZ-Expansion-AI shooter (hostile Raider with an SVD) at 50/100/200/400 m; every server-side shot is logged by
	//     the lab mod's Weapon_Base.EEFired hook (needs -Expansion: @CF, @DayZ-Expansion-Core, @DayZ-Expansion-AI).
	// AZ_LAB_MODE_VALUE is replaced with the number by the build script.
	static const int AZ_LAB_MODE = AZ_LAB_MODE_VALUE;
	// Weather over any lab mode, so labelled gunfire can be heard in it: 0 none, 1 storm in heavy rain, 2 dry storm, 3 heavy rain with no storm. Replaced by the build script.
	static const int AZ_LAB_WEATHER = AZ_LAB_WEATHER_VALUE;
	// Mode 34: the hour to set, -1 leaves the mission's own time and weather.
	static const int AZ_LAB_HOUR = AZ_LAB_HOUR_VALUE;
	// Mode 34, -Screens: a map to open, and the character dies at this many seconds after the warmup, for the death screen. 0 = off.
	static const int AZ_LAB_DIE_AT = AZ_LAB_DIE_AT_VALUE;
	protected float m_AzOverlayClock;
	protected bool m_AzOverlayDone;

	// ---- scripted view turns (mode 2): ground truth for the video yaw tracker with nobody at the controls ----
	// The server overrides the player's horizontal aim change, the same call the game's own developer "server walk"
	// debug uses (OverrideAimChangeX). Segments of AZ_TURN_SECONDS at a rate, separated by still pauses; each segment
	// start/end is logged to assistz_turns_<utc>.csv with game_ms so it lines up with the pose log exactly.
	static const float AZ_TURN_SECONDS = 2.0;
	static const float AZ_TURN_PAUSE = 2.0;
	static const float AZ_TURN_WARMUP = 25.0;
	protected string m_AzTurnPath;
	protected float m_AzTurnClock;
	protected int m_AzTurnIndex;
	protected bool m_AzTurning;

	// ---- world-fixed beacons (mode 3) ----
	// When the first one fires, two world positions are fixed relative to the body: A at bearing 40 (front-right) and B at
	// 250 (behind-left; at 220 the two folded to +-7 deg under a 33 deg free-look and merged), both AZ_FIXED_DISTANCE m away. They then alternate every AZ_FIXED_INTERVAL s, so each source
	// repeats every 2 x interval s while the view turns; the true head-relative bearing of every detonation follows from
	// the pose log. Logged to the beacon CSV like mode 1 (bearing_rel = the source's bearing at the moment it fires).
	static const float AZ_FIXED_INTERVAL = 1.5;
	static const float AZ_FIXED_DISTANCE = 60.0;
	protected bool m_AzFixedSet;
	protected vector m_AzFixedA;
	protected vector m_AzFixedB;
	protected bool m_AzFixedNextIsA = true;

	// ---- lab sniper (mode 4) ----
	// An SVD spawned AZ_SNIPER bearing 0 (straight ahead of the body when the shot is planned) at each range of the ladder,
	// chambered (FillChamber) and fired with Weapon.Fire toward a point AZ_MISS_OFFSET m to the player's right at head
	// height: a near miss, so the supersonic crack (sonicCrack_* sound sets) reaches the player before the muzzle report.
	// 4 shots per range, AZ_SNIPER_INTERVAL s apart: 2 NEAR misses (AZ_MISS_OFFSET m from the head) then 2 WIDE misses
	// (AZ_WIDE_OFFSET m) - TJ: the game plays a special noise when a bullet passes close that you do not hear when it
	// lands far away (the bullet_flyby / sonicCrack_Close sets), i.e. a "you are being shot at" cue. Each shot logged
	// to assistz_shots_<utc>.csv with game_ms and its miss distance. Expected crack-to-report gap on near misses
	// ~ d * (1/343 - 1/830) s for 7.62x54R (~1.7 ms per metre).
	static const float AZ_SNIPER_INTERVAL = 3.0;
	static const float AZ_MISS_OFFSET = 2.0;
	static const float AZ_WIDE_OFFSET = 25.0;
	protected string m_AzShotPath;
	protected int m_AzShotIndex;
	// The shooter: a dummy survivor spawned like the developer console does (CreateObjectEx + OnSpawnedFromConsole), so
	// the rifle has an OWNER. Weapon.Fire on an unowned SVD lying in the world crashed the server (2026-11-03: access
	// violation reading 0x238 in AzSniperTick); the same call on a weapon in a character's hands is the attempt now.
	protected PlayerBase m_AzShooter;
	protected float m_AzShooterRange;
	// A fresh rifle before EVERY shot: the client copy's weapon state machine fires the first shot of a new rifle and
	// then desyncs (lab 2026-11-03: one heard shot per rifle at 100/200/400 m), so each shot is a first shot.
	protected bool m_AzArmed;
	// Where the dummy should point (beside the listener's head). The server simulates the dummy, so its aim replicates to
	// the client copy that fires: AzShooterAimHold steers the dummy's weapon pitch onto it every update.
	protected vector m_AzAimPoint;
	protected bool m_AzHasAimPoint;

	// ---- sound beacon: directional ground truth ----
	// Every AZ_BEACON_INTERVAL s a fragmentation grenade is detonated at the next (bearing, distance) of a 16-shot plan,
	// bearing RELATIVE TO THE BODY'S FORWARD (0 = ahead, 90 = right, 180 = behind, 270 = left). Shots 0-7: front/back
	// mirror pairs at 50 m (45/135, 315/225, 90/270, 0/180 fold to the same stereo azimuth, so any difference between a
	// pair is a front/back cue). Shots 8-15: distance ladders 35/70/140/280 m at 90 and at 0 degrees (level falloff).
	// Each detonation is logged to assistz_beacon_<utc>.csv at the moment it is triggered.
	static const float AZ_BEACON_INTERVAL = 6.0;
	static const float AZ_BEACON_WARMUP = 15.0;
	protected string m_AzBeaconPath;
	protected float m_AzBeaconAccum;
	protected float m_AzBeaconWarmup;
	protected int m_AzBeaconIndex;

	override void OnMissionStart()
	{
		super.OnMissionStart();
		int year, month, day, hour, minute, second;
		GetYearMonthDayUTC(year, month, day);
		GetHourMinuteSecondUTC(hour, minute, second);
		string stamp = string.Format("%1%2%3_%4%5%6", year, month.ToStringLen(2), day.ToStringLen(2), hour.ToStringLen(2), minute.ToStringLen(2), second.ToStringLen(2));
		m_AzPosePath = "$profile:assistz_pose_" + stamp + ".csv";
		FileHandle file = OpenFile(m_AzPosePath, FileMode.WRITE);
		if (file)
		{
			m_AzPoseOpen = true;
			FPrintln(file, "game_ms,utc,x,y,z,yaw,pitch,roll,aim_lr,aim_ud");
			CloseFile(file);
		}
		Print("[AssistZ] pose log " + stamp + " open=" + m_AzPoseOpen.ToString());
		m_AzTurnPath = "$profile:assistz_turns_" + stamp + ".csv";
		if (AZ_LAB_MODE == 2)
		{
			FileHandle turns = OpenFile(m_AzTurnPath, FileMode.WRITE);
			if (turns)
			{
				FPrintln(turns, "game_ms,utc,event,segment,aim_change");
				CloseFile(turns);
			}
		}
		Print("[AssistZ] lab mode " + AZ_LAB_MODE.ToString());
		m_AzShotPath = "$profile:assistz_shots_" + stamp + ".csv";
		if (AZ_LAB_MODE == 4 || AZ_LAB_MODE == 5 || AZ_LAB_MODE == 29 || AZ_LAB_MODE == 33)
		{
			FileHandle shots = OpenFile(m_AzShotPath, FileMode.WRITE);
			if (shots)
			{
				FPrintln(shots, "game_ms,utc,shot,distance,miss,fired,src_x,src_y,src_z,x,y,z,yaw,aim_lr,weapon");
				CloseFile(shots);
			}
		}
		m_AzEnvPath = "$profile:assistz_env_" + stamp + ".csv";
		FileHandle envFile = OpenFile(m_AzEnvPath, FileMode.WRITE);
		if (envFile)
		{
			FPrintln(envFile, "game_ms,overcast,rain,fog,wind,infected,infected_nearest,animals,animal_nearest,vehicles,vehicle_nearest,infected_state,infected_worst,infected_x,infected_z,placed_x,placed_z,placed_state,placed_y,placed_los");
			CloseFile(envFile);
		}
		m_AzAIPosePath = "$profile:assistz_aipose_" + stamp + ".csv";
		if (AZ_LAB_MODE == 6 || AZ_LAB_MODE == 10 || AzUsesDecoy())
		{
			FileHandle aipose = OpenFile(m_AzAIPosePath, FileMode.WRITE);
			if (aipose)
			{
				FPrintln(aipose, "game_ms,range,ai,x,y,z,speed,dist,state");
				CloseFile(aipose);
			}
		}
		m_AzBeaconPath = "$profile:assistz_beacon_" + stamp + ".csv";
		FileHandle beacon = OpenFile(m_AzBeaconPath, FileMode.WRITE);
		if (beacon)
		{
			FPrintln(beacon, "game_ms,utc,bearing_rel,distance,src_x,src_y,src_z,x,y,z,yaw,aim_lr,fwd_x,fwd_z");
			CloseFile(beacon);
		}
	}

	override void OnUpdate(float timeslice)
	{
		super.OnUpdate(timeslice);
		if (!m_AzPoseOpen)
			return;
		m_AzPoseFlushAccum += timeslice;
		if (m_AzPoseFlushAccum >= 1.0)
		{
			m_AzPoseFlushAccum = 0;
			AzPoseFlush();
		}
		m_AzPoseAccum += timeslice;
		if (m_AzPoseAccum < 0.1)
			return;
		m_AzPoseAccum = 0;
		m_AzPlayers.Clear();
		GetGame().GetPlayers(m_AzPlayers);
		if (m_AzPlayers.Count() == 0)
			return;
		PlayerBase player = AzFindPlayer("AssistZLab");
		if (!player)
			player = PlayerBase.Cast(m_AzPlayers[0]);
		if (!player)
			return;
		if (AZ_LAB_MODE == 1)
			AzBeaconTick(player, 0.1);
		if (AZ_LAB_MODE == 2)
			AzTurnTick(player, 0.1);
		if (AZ_LAB_MODE == 3)
			AzFixedTick(player, 0.1);
		if (AZ_LAB_MODE == 4 || AZ_LAB_MODE == 29 || AZ_LAB_MODE == 33)
		{
			AzSniperTick(player, 0.1);
			AzShooterAimHold();
		}
		if (AZ_LAB_MODE == 5)
			AzRealShooterTick(player, 0.1);
		if (AZ_LAB_MODE == 6 && !m_AzMode6Seen)
		{
			m_AzMode6Seen = true;
			Print("[AssistZ] mode 6 dispatch reached");
		}
		AzEnvTick(player, 0.1);
		if (AZ_LAB_WEATHER != 0 && !m_AzOverlayDone)
		{
			m_AzOverlayClock += 0.1;
			if (m_AzOverlayClock >= 5.0)
			{
				m_AzOverlayDone = true;
				float overlayRain = 1.0;
				if (AZ_LAB_WEATHER == 2)
					overlayRain = 0;
				if (AZ_LAB_WEATHER == 3)
				{
					AzWeatherSet(1.0, 1.0, 0, 15);
					AzWorldLog("rain", 0, player.GetPosition());
				}
				else
				{
					AzWeatherSet(1.0, overlayRain, 1.0, 15);
					AzWorldLog("storm", 15, player.GetPosition());
				}
			}
		}
		if (AZ_LAB_MODE == 11)
			AzWorldEventTick(player, 0.1);
		if (AZ_LAB_MODE == 12)
			AzWeatherLabTick(player, 0.1);
		if (AZ_LAB_MODE == 13 || AZ_LAB_MODE == 14 || AZ_LAB_MODE == 15 || AZ_LAB_MODE == 24)
			AzStormLabTick(player, 0.1);
		if (AZ_LAB_MODE == 16 || AZ_LAB_MODE == 17 || AZ_LAB_MODE == 18 || AZ_LAB_MODE == 21)
			AzInfectedLabTick(player, 0.1);
		if (AZ_LAB_MODE == 19 || AZ_LAB_MODE == 20 || AZ_LAB_MODE == 22 || AZ_LAB_MODE == 23)
			AzSurveyTick(player, 0.1);
		if (AZ_LAB_MODE == 30)
			AzInfectedStatesTick(player, 0.1);
		if (AZ_LAB_MODE == 31)
			AzCalmInfectedTick(player, 0.1);
		if (AZ_LAB_MODE == 32)
			AzOwnSoundsTick(player, 0.1);
		if (AZ_LAB_MODE == 34)
			AzSeeInfectedTick(player, 0.1);
		if (AZ_LAB_MODE == 26 || AZ_LAB_MODE == 27)
			AzMapSurveyTick(player, 0.1);
		// No #ifdef here: init.c does not see the mods' defines; the Expansion calls are in the lab mod (AssistZLabExpansion).
		if (AZ_LAB_MODE == 6 || AZ_LAB_MODE == 10 || AzUsesDecoy())
			AzExpansionTick(player, 0.1);
		vector pos = player.GetPosition();
		vector ori = player.GetOrientation();
		float aimLR = 0;
		float aimUD = 0;
		HumanCommandWeapons hcw = player.GetCommandModifier_Weapons();
		if (hcw)
		{
			aimLR = hcw.GetBaseAimingAngleLR();
			aimUD = hcw.GetBaseAimingAngleUD();
		}
		int hour, minute, second;
		GetHourMinuteSecondUTC(hour, minute, second);
		string utc = hour.ToStringLen(2) + ":" + minute.ToStringLen(2) + ":" + second.ToStringLen(2);
		string line = GetGame().GetTime().ToString() + "," + utc;
		line += "," + pos[0].ToString() + "," + pos[1].ToString() + "," + pos[2].ToString();
		line += "," + ori[0].ToString() + "," + ori[1].ToString() + "," + ori[2].ToString();
		line += "," + aimLR.ToString() + "," + aimUD.ToString();
		m_AzPoseRows.Insert(line);
	}

	// Once a second: the weather state, and the living infected, animals and vehicles within AZ_ENV_RADIUS of the listener. Ground truth for
	// CLAP's thunder, rain, wind and zombie words. The script API has weather state but no lightning event, so thunder can only be
	// matched to conditions, not to a strike.
	static const float AZ_ENV_RADIUS = 200.0;
	protected string m_AzEnvPath;
	protected float m_AzEnvClock;

	protected void AzEnvTick(PlayerBase player, float dt)
	{
		m_AzEnvClock += dt;
		if (m_AzEnvClock < 1.0)
			return;
		m_AzEnvClock = 0;
		Weather weather = GetGame().GetWeather();
		vector at = player.GetPosition();
		array<Object> near = new array<Object>;
		GetGame().GetObjectsAtPosition(at, AZ_ENV_RADIUS, near, null);
		int infected = 0;
		int animals = 0;
		int vehicles = 0;
		float infectedNearest = -1;
		float animalNearest = -1;
		float vehicleNearest = -1;
		// The nearest infected's mind state, from the game: 0 calm, 1 disturbed, 2 alerted, 3 chase, 4 fight, 5 cower (DayZInfectedConstants, in the order of its enum). -1 none.
		int infectedState = -1;
		// The most worked up of any infected in range, so a calm nearest one cannot hide an agitated one behind it (not counting cower).
		int infectedWorst = -1;
		vector infectedAt = "0 0 0";
		foreach (Object o : near)
		{
			float d = vector.Distance(o.GetPosition(), at);
			if (o.IsInherited(ZombieBase) && EntityAI.Cast(o).IsAlive())
			{
				// Mode 34 scores pictures against one known infected: a natural one in view would count as a false alarm, and could be the "nearest" that infected_x,z reports.
				if (AZ_LAB_MODE == 34 && o != m_AzInfected)
				{
					GetGame().ObjectDelete(o);
					continue;
				}
				infected++;
				ZombieBase anyInfected = ZombieBase.Cast(o);
				int anyState = anyInfected.GetMindStateSynced() - DayZInfectedConstants.MINDSTATE_CALM;
				if (anyState > infectedWorst && anyState <= 4) infectedWorst = anyState;
				if (infectedNearest < 0 || d < infectedNearest)
				{
					infectedNearest = d;
					infectedAt = o.GetPosition();
					ZombieBase infectedBody = ZombieBase.Cast(o);
					infectedState = infectedBody.GetMindStateSynced() - DayZInfectedConstants.MINDSTATE_CALM;
				}
			}
			else if (o.IsInherited(AnimalBase) && EntityAI.Cast(o).IsAlive())
			{
				animals++;
				if (animalNearest < 0 || d < animalNearest) animalNearest = d;
			}
			else if (o.IsInherited(Transport))
			{
				vehicles++;
				if (vehicleNearest < 0 || d < vehicleNearest) vehicleNearest = d;
			}
		}
		string envLine = GetGame().GetTime().ToString() + "," + weather.GetOvercast().GetActual().ToString() + "," + weather.GetRain().GetActual().ToString();
		envLine += "," + weather.GetFog().GetActual().ToString() + "," + weather.GetWindSpeed().ToString();
		// Split: one long line is a "Formula too complex" compile error.
		envLine += "," + infected.ToString() + "," + infectedNearest.ToString() + "," + animals.ToString() + "," + animalNearest.ToString();
		envLine += "," + vehicles.ToString() + "," + vehicleNearest.ToString() + "," + infectedState.ToString() + "," + infectedWorst.ToString();
		envLine += "," + infectedAt[0].ToString() + "," + infectedAt[2].ToString();
		// The one the lab placed, wherever it has wandered, since the nearest of all can be another.
		vector placedAt = "0 0 0";
		int placedState = -1;
		// How many of three rays from the player's eye reach it (knee, waist, head), so a hill or a bush in the way does not count as "in view". -1 none out.
		int placedLos = -1;
		ZombieBase placedBody = ZombieBase.Cast(m_AzInfected);
		if (placedBody && placedBody.IsAlive())
		{
			placedAt = placedBody.GetPosition();
			placedState = placedBody.GetMindStateSynced() - DayZInfectedConstants.MINDSTATE_CALM;
			placedLos = 0;
			vector losFrom = at;
			losFrom[1] = at[1] + 1.6;
			vector losTo;
			vector losHit;
			vector losDir;
			int losComponent;
			bool losBlocked;
			for (int losStep = 0; losStep < 3; losStep++)
			{
				losTo = placedAt;
				losTo[1] = placedAt[1] + 0.4 + 0.6 * losStep;
				losBlocked = DayZPhysics.RaycastRV(losFrom, losTo, losHit, losDir, losComponent, null, null, player, false, false, ObjIntersectView);
				if (!losBlocked || vector.Distance(losHit, losTo) < 1.0)
					placedLos++;
			}
		}
		envLine += "," + placedAt[0].ToString() + "," + placedAt[2].ToString() + "," + placedState.ToString() + "," + placedAt[1].ToString();
		envLine += "," + placedLos.ToString();
		FileHandle file = OpenFile(m_AzEnvPath, FileMode.APPEND);
		if (file)
		{
			FPrintln(file, envLine);
			CloseFile(file);
		}
	}

	// Mode 11: world events at known times and distances, so each sound gets a label from the game. Slot i = type i / 3 (heli crash, sleigh crash, gas strike), distance i % 3 (300, 1000, 2500 m). The game
	// only runs these sounds through EEOnCECreate when its economy spawns the object, so the lab calls the same code: the crash's RequestSoundEvent, or the gas sequence.
	// A gas strike's Artillery_Close arrives minutes after the trigger: 182-208 s on run sniper_20261104_051244, matching the game's delay (distance from the firing position / 100 + 20 s). With a minute between
	// events it landed on the crashes, so crashes go first, 45 s apart, and strikes 220 s apart. Crash sounds carry the speed of sound: sleigh +14.0 s at 318 m, +20.4 s at 2501 m.
	static const float AZ_WORLD_CRASH_SECONDS = 45.0;
	static const float AZ_WORLD_GAS_SECONDS = 220.0;
	static const int AZ_WORLD_EVENT_COUNT = 9;
	protected float m_AzWorldClock;
	protected int m_AzWorldIndex;
	protected string m_AzWorldPath;

	protected void AzWorldEventTick(PlayerBase player, float dt)
	{
		if (m_AzBeaconWarmup < AZ_BEACON_WARMUP)
		{
			m_AzBeaconWarmup += dt;
			return;
		}
		if (m_AzWorldIndex >= AZ_WORLD_EVENT_COUNT)
			return;
		m_AzWorldClock += dt;
		float wait = AZ_WORLD_CRASH_SECONDS;
		if (m_AzWorldIndex > 6)
			wait = AZ_WORLD_GAS_SECONDS;
		if (m_AzWorldIndex > 0 && m_AzWorldClock < wait)
			return;
		m_AzWorldClock = 0;
		int type = m_AzWorldIndex / 3;
		int slot = m_AzWorldIndex - type * 3;
		float distance = 300;
		if (slot == 1) distance = 1000;
		if (slot == 2) distance = 2500;
		vector at = player.GetPosition();
		vector fwd = player.GetDirection();
		fwd[1] = 0;
		fwd.Normalize();
		// The first bearing on dry land.
		vector spot = at;
		for (int tryIndex = 0; tryIndex < 12; tryIndex++)
		{
			vector candidate = AzOffset(at, fwd, tryIndex * 30.0 + m_AzWorldIndex * 41.0, distance);
			if (!GetGame().SurfaceIsSea(candidate[0], candidate[2]) && !GetGame().SurfaceIsPond(candidate[0], candidate[2]))
			{
				spot = candidate;
				break;
			}
		}
		spot[1] = GetGame().SurfaceY(spot[0], spot[2]);
		string name = "gas";
		if (type == 2)
		{
			EntityAI area = EntityAI.Cast(GetGame().CreateObject("ContaminatedArea_Dynamic", spot));
			if (area)
				area.EEOnCECreate();
			else
				name = "gas_failed";
		}
		else
		{
			string wreck = "Wreck_UH1Y";
			name = "heli";
			if (type == 1)
			{
				wreck = "Wreck_SantasSleigh";
				name = "sleigh";
			}
			CrashBase crash = CrashBase.Cast(GetGame().CreateObject(wreck, spot));
			if (crash)
				crash.RequestSoundEvent();
			else
				name = name + "_failed";
		}
		AzWorldLog(name, vector.Distance(spot, at), spot);
		m_AzWorldIndex++;
	}

	// Mode 12: ground truth for CLAP's weather and infected words, which the natural runs never tested (all dry, infected no closer than 107 m). Phases of AZ_WEATHER_PHASE_SECONDS: clear, heavy rain, thunderstorm; then clear again with one
	// infected at 10, 25 and 50 m, AZ_INFECTED_SECONDS each. MissionWeather(true) stops the world script overriding this. The script API has no lightning event: thunder is known only by the storm being on. Phases go to world.csv, weather values to env.csv.
	static const float AZ_WEATHER_PHASE_SECONDS = 150.0;
	static const float AZ_INFECTED_SECONDS = 45.0;
	protected float m_AzWeatherClock;
	protected int m_AzWeatherPhase;
	protected Object m_AzInfected;

	protected void AzWeatherSet(float overcast, float rain, float storm, float stormTimeout)
	{
		Weather weather = GetGame().GetWeather();
		weather.MissionWeather(true);
		weather.SetRainThresholds(0, 1, 0);
		weather.GetOvercast().Set(overcast, 5, 1000);
		weather.GetRain().Set(rain, 5, 1000);
		weather.GetFog().Set(0, 5, 1000);
		weather.SetStorm(storm, 0, stormTimeout);
	}

	protected void AzWorldLog(string name, float distance, vector spot)
	{
		if (m_AzWorldPath == "")
		{
			m_AzWorldPath = "$profile:assistz_world_" + GetGame().GetTime().ToString() + ".csv";
			FileHandle head = OpenFile(m_AzWorldPath, FileMode.WRITE);
			if (head)
			{
				FPrintln(head, "game_ms,event,distance,x,y,z");
				CloseFile(head);
			}
		}
		FileHandle file = OpenFile(m_AzWorldPath, FileMode.APPEND);
		if (file)
		{
			FPrintln(file, GetGame().GetTime().ToString() + "," + name + "," + distance.ToString() + "," + spot[0].ToString() + "," + spot[1].ToString() + "," + spot[2].ToString());
			CloseFile(file);
		}
		Print("[AssistZ] world event " + name + " at " + distance.ToString() + " m");
	}

	protected void AzWeatherLabTick(PlayerBase player, float dt)
	{
		if (m_AzBeaconWarmup < AZ_BEACON_WARMUP)
		{
			m_AzBeaconWarmup += dt;
			return;
		}
		player.SetAllowDamage(false);
		m_AzWeatherClock += dt;
		float wait = AZ_WEATHER_PHASE_SECONDS;
		if (m_AzWeatherPhase > 3)
			wait = AZ_INFECTED_SECONDS;
		if (m_AzWeatherPhase > 0 && m_AzWeatherClock < wait)
			return;
		if (m_AzWeatherPhase > 6)
			return;
		m_AzWeatherClock = 0;
		vector at = player.GetPosition();
		if (m_AzInfected)
		{
			GetGame().ObjectDelete(m_AzInfected);
			m_AzInfected = null;
		}
		if (m_AzWeatherPhase == 0)
		{
			AzWeatherSet(0.1, 0, 0, 15);
			AzWorldLog("clear", 0, at);
		}
		else if (m_AzWeatherPhase == 1)
		{
			AzWeatherSet(0.8, 0.9, 0, 15);
			AzWorldLog("rain", 0, at);
		}
		else if (m_AzWeatherPhase == 2)
		{
			AzWeatherSet(1.0, 1.0, 1.0, 15);
			AzWorldLog("storm", 0, at);
		}
		else if (m_AzWeatherPhase == 3)
		{
			AzWeatherSet(0.1, 0, 0, 15);
			AzWorldLog("clear_after", 0, at);
		}
		else
		{
			float distance = 10;
			if (m_AzWeatherPhase == 5) distance = 25;
			if (m_AzWeatherPhase == 6) distance = 50;
			vector fwd = player.GetDirection();
			fwd[1] = 0;
			fwd.Normalize();
			vector spot = AzOffset(at, fwd, m_AzWeatherPhase * 70.0, distance);
			spot[1] = GetGame().SurfaceY(spot[0], spot[2]);
			m_AzInfected = GetGame().CreateObject("ZmbM_CitizenASkinny_Blue", spot, false, true, true);
			string name = "infected";
			if (!m_AzInfected)
				name = "infected_failed";
			AzWorldLog(name, distance, spot);
		}
		m_AzWeatherPhase++;
	}

	// Modes 13 and 14: is what the storm adds to the audio thunder? The script API has no lightning event, but SetStorm takes a minimum time between strikes, so its spacing must follow the setting. Clear, then two storm phases with that time at 45 and 15 s (mode 13) or
	// 15 and 45 s (mode 14), then the same rain with no storm. world.csv logs the time in the distance column. -RecordSeconds 800. Mode 15 tests settings not used to form the burst definition: 90 s for 500 s, 30 s for 300 s; -RecordSeconds 1000.
	static const float AZ_STORM_CLEAR_SECONDS = 60.0;
	static const float AZ_STORM_PHASE_SECONDS = 290.0;
	static const float AZ_STORM_RAIN_SECONDS = 120.0;

	protected void AzStormLabTick(PlayerBase player, float dt)
	{
		if (m_AzBeaconWarmup < AZ_BEACON_WARMUP)
		{
			m_AzBeaconWarmup += dt;
			return;
		}
		player.SetAllowDamage(false);
		m_AzWeatherClock += dt;
		float wait = AZ_STORM_PHASE_SECONDS;
		if (AZ_LAB_MODE == 15 && m_AzWeatherPhase == 2)
			wait = 500;
		if (AZ_LAB_MODE == 15 && m_AzWeatherPhase == 3)
			wait = 300;
		if (m_AzWeatherPhase == 1)
			wait = AZ_STORM_CLEAR_SECONDS;
		if (m_AzWeatherPhase > 0 && m_AzWeatherClock < wait)
			return;
		if (m_AzWeatherPhase > 3)
			return;
		m_AzWeatherClock = 0;
		vector at = player.GetPosition();
		float first = 45;
		float second = 15;
		if (AZ_LAB_MODE == 14)
		{
			first = 15;
			second = 45;
		}
		if (AZ_LAB_MODE == 15)
		{
			first = 90;
			second = 30;
		}
		// Mode 24: thunder with no rain. Nasdara's storm (rain 0.3) read as heli crash and gunshot: the class had only heard thunder under heavy rain. 15 then 30 s, then the same overcast with no storm. -RecordSeconds 800.
		float stormRain = 1.0;
		if (AZ_LAB_MODE == 24)
		{
			first = 15;
			second = 30;
			stormRain = 0;
		}
		if (m_AzWeatherPhase == 0)
		{
			AzWeatherSet(0.1, 0, 0, 15);
			AzWorldLog("clear", 0, at);
		}
		else if (m_AzWeatherPhase == 1)
		{
			AzWeatherSet(1.0, stormRain, 1.0, first);
			AzWorldLog("storm", first, at);
		}
		else if (m_AzWeatherPhase == 2)
		{
			AzWeatherSet(1.0, stormRain, 1.0, second);
			AzWorldLog("storm", second, at);
		}
		else
		{
			AzWeatherSet(1.0, stormRain, 0, 15);
			AzWorldLog("rain", 0, at);
		}
		m_AzWeatherPhase++;
	}

	// Modes 16 and 17: infected attacks at known times, for a word the probe has never trained on: mode 14's natural attack was called thunder. Clear weather; six infected, one at a time, AZ_ATTACK_SPAWN_M from the invulnerable listener for 45 s with 20 s between; then 120 s of storm and a clear tail.
	// Mode 17 uses other types, so what a probe trained on mode 16 meets is new. world.csv logs each spawn; env.csv's nearest infected shows whether it attacked. -RecordSeconds 600.
	static const float AZ_ATTACK_SPAWN_M = 2.0;

	protected string AzAttackType(int index)
	{
		if (AZ_LAB_MODE == 16 || AZ_LAB_MODE == 18)
		{
			if (index == 0) return "ZmbM_CitizenASkinny_Blue";
			if (index == 1) return "ZmbF_CitizenANormal_Blue";
			if (index == 2) return "ZmbM_SoldierNormal";
			if (index == 3) return "ZmbF_JournalistNormal_Blue";
			if (index == 4) return "ZmbM_FirefighterNormal";
			return "ZmbF_BlueCollarFat_Blue";
		}
		if (index == 0) return "ZmbM_HunterOld_Autumn";
		if (index == 1) return "ZmbF_SurvivorNormal_Blue";
		if (index == 2) return "ZmbM_PatrolNormal_Autumn";
		if (index == 3) return "ZmbF_Clerk_Normal_Blue";
		if (index == 4) return "ZmbM_ClerkFat_Brown";
		return "ZmbF_MechanicNormal_Beige";
	}

	// Mode 30: what does an infected sound like in each mind state? One at a time, 12 types, moving round the invulnerable listener and through 4, 10, 20, 40 m, for 40 s with 10 s between, so env.csv's infected_state and infected_nearest give each sound its truth. Near ones notice and attack, far ones mostly do not. -RecordSeconds 640.
	static const float AZ_STATES_SECONDS = 40.0;
	static const float AZ_STATES_REST = 10.0;
	static const int AZ_STATES_SLOTS = 12;
	protected int m_AzStatesSlot;
	protected bool m_AzStatesOut;

	protected float AzStatesDistance(int slot)
	{
		int step = slot - (slot / 4) * 4;
		if (step == 0) return 4;
		if (step == 1) return 10;
		if (step == 2) return 20;
		return 40;
	}

	protected string AzStatesType(int slot)
	{
		if (slot == 0) return "ZmbM_CitizenASkinny_Blue";
		if (slot == 1) return "ZmbF_CitizenANormal_Blue";
		if (slot == 2) return "ZmbM_SoldierNormal";
		if (slot == 3) return "ZmbF_JournalistNormal_Blue";
		if (slot == 4) return "ZmbM_FirefighterNormal";
		if (slot == 5) return "ZmbF_BlueCollarFat_Blue";
		if (slot == 6) return "ZmbM_HunterOld_Autumn";
		if (slot == 7) return "ZmbF_SurvivorNormal_Blue";
		if (slot == 8) return "ZmbM_PatrolNormal_Autumn";
		if (slot == 9) return "ZmbF_Clerk_Normal_Blue";
		if (slot == 10) return "ZmbM_ClerkFat_Brown";
		return "ZmbF_MechanicNormal_Beige";
	}

	protected void AzInfectedStatesTick(PlayerBase player, float dt)
	{
		if (m_AzBeaconWarmup < AZ_BEACON_WARMUP)
		{
			m_AzBeaconWarmup += dt;
			return;
		}
		player.SetAllowDamage(false);
		m_AzWeatherClock += dt;
		vector at = player.GetPosition();
		if (m_AzStatesOut)
		{
			if (m_AzWeatherClock < AZ_STATES_SECONDS)
				return;
			m_AzWeatherClock = 0;
			m_AzStatesOut = false;
			if (m_AzInfected)
			{
				GetGame().ObjectDelete(m_AzInfected);
				m_AzInfected = null;
			}
			AzWorldLog("rest", 0, at);
			return;
		}
		if (m_AzStatesSlot >= AZ_STATES_SLOTS || m_AzWeatherClock < AZ_STATES_REST)
			return;
		m_AzWeatherClock = 0;
		vector fwd = player.GetDirection();
		fwd[1] = 0;
		fwd.Normalize();
		float dist = AzStatesDistance(m_AzStatesSlot);
		vector spot = AzOffset(at, fwd, 75.0 * m_AzStatesSlot, dist);
		spot[1] = GetGame().SurfaceY(spot[0], spot[2]);
		string type = AzStatesType(m_AzStatesSlot);
		m_AzInfected = GetGame().CreateObject(type, spot, false, true, true);
		if (m_AzInfected)
			AzWorldLog("infected", dist, spot);
		else
			AzWorldLog("infected_failed", dist, spot);
		Print("[AssistZ] mode 30 slot " + m_AzStatesSlot.ToString() + " type " + type + " at " + dist.ToString() + " m");
		m_AzStatesSlot++;
		m_AzStatesOut = true;
	}

	// Mode 31: calm infected make few sounds, so three at once, 60 s out and 20 s with none, 12 slots, at 15, 25 and 40 m and turning round the invulnerable listener. env.csv's infected_worst says whether any of them was worked up. -RecordSeconds 1000.
	static const float AZ_CALM_SECONDS = 60.0;
	static const float AZ_CALM_REST = 20.0;
	protected ref array<Object> m_AzCalmGroup = new array<Object>;

	protected void AzCalmInfectedTick(PlayerBase player, float dt)
	{
		if (m_AzBeaconWarmup < AZ_BEACON_WARMUP)
		{
			m_AzBeaconWarmup += dt;
			return;
		}
		player.SetAllowDamage(false);
		m_AzWeatherClock += dt;
		vector at = player.GetPosition();
		if (m_AzStatesOut)
		{
			if (m_AzWeatherClock < AZ_CALM_SECONDS)
				return;
			m_AzWeatherClock = 0;
			m_AzStatesOut = false;
			foreach (Object old : m_AzCalmGroup)
			{
				if (old)
					GetGame().ObjectDelete(old);
			}
			m_AzCalmGroup.Clear();
			AzWorldLog("rest", 0, at);
			return;
		}
		if (m_AzStatesSlot >= AZ_STATES_SLOTS || m_AzWeatherClock < AZ_CALM_REST)
			return;
		m_AzWeatherClock = 0;
		vector fwd = player.GetDirection();
		fwd[1] = 0;
		fwd.Normalize();
		float bearing = 75.0 * m_AzStatesSlot;
		for (int member = 0; member < 3; member++)
		{
			float memberDist = 15;
			if (member == 1) memberDist = 25;
			if (member == 2) memberDist = 40;
			vector spot = AzOffset(at, fwd, bearing + member * 120.0, memberDist);
			spot[1] = GetGame().SurfaceY(spot[0], spot[2]);
			int typeIndex = m_AzStatesSlot + member * 4;
			typeIndex = typeIndex - (typeIndex / 12) * 12;
			Object body = GetGame().CreateObject(AzStatesType(typeIndex), spot, false, true, true);
			if (body)
				m_AzCalmGroup.Insert(body);
		}
		AzWorldLog("infected", 25, at);
		m_AzStatesSlot++;
		m_AzStatesOut = true;
	}

	// Mode 34: what can video see? One infected at a time, in view ahead of the invulnerable listener, placed at 10, 20, 40, 60, 80, 120 m and within 25 degrees of straight ahead, 40 s each with 20 s of nothing between, 12 types. They wander from there. env.csv gives where it is and its state once a second (placed_x, placed_z, placed_state); natural infected are removed; frames come from -FrameDir. -RecordSeconds 800.
	static const float AZ_SEE_SECONDS = 40.0;
	static const float AZ_SEE_REST = 20.0;
	protected bool m_AzSeeLit;
	protected vector m_AzSeeHome;
	protected float m_AzSeeClock;
	protected bool m_AzSeeDead;

	protected float AzSeeDistance(int slot)
	{
		int step = slot - (slot / 6) * 6;
		if (step == 0) return 10;
		if (step == 1) return 20;
		if (step == 2) return 40;
		if (step == 3) return 60;
		if (step == 4) return 80;
		return 120;
	}

	protected void AzSeeInfectedTick(PlayerBase player, float dt)
	{
		if (m_AzBeaconWarmup < AZ_BEACON_WARMUP)
		{
			m_AzBeaconWarmup += dt;
			return;
		}
		player.SetAllowDamage(false);
		m_AzWeatherClock += dt;
		vector at = player.GetPosition();
		if (!m_AzSeeLit)
		{
			m_AzSeeLit = true;
			if (AZ_LAB_HOUR >= 0)
			{
				GetGame().GetWorld().SetDate(2026, 6, 15, AZ_LAB_HOUR, 0);
				if (AZ_LAB_WEATHER == 0)
					AzWeatherSet(0.1, 0, 0, 15);
			}
			AzWorldLog("hour", AZ_LAB_HOUR, at);
			m_AzSeeHome = at;
			if (AZ_LAB_DIE_AT > 0)
				player.GetInventory().CreateInInventory("ChernarusMap");
		}
		m_AzSeeClock += dt;
		if (AZ_LAB_DIE_AT > 0 && !m_AzSeeDead && m_AzSeeClock >= AZ_LAB_DIE_AT)
		{
			m_AzSeeDead = true;
			player.SetAllowDamage(true);
			player.SetHealth("", "", 0);
			AzWorldLog("die", 0, at);
			return;
		}
		if (m_AzSeeDead)
			return;
		// A driven player drifts: sprinting forward beats walking back. Runs went up to 300 m and several into the sea (TJ saw one swimming, 2026-11-06). Back home on water at once, or between slots if far off.
		bool azInWater = GetGame().SurfaceIsSea(at[0], at[2]) || GetGame().SurfaceIsPond(at[0], at[2]);
		if (azInWater || (!m_AzStatesOut && vector.Distance(at, m_AzSeeHome) > 40.0))
		{
			player.SetPosition(m_AzSeeHome);
			AzWorldLog("home", 0, m_AzSeeHome);
			at = m_AzSeeHome;
		}
		if (m_AzStatesOut)
		{
			if (m_AzWeatherClock < AZ_SEE_SECONDS)
				return;
			m_AzWeatherClock = 0;
			m_AzStatesOut = false;
			if (m_AzInfected)
			{
				GetGame().ObjectDelete(m_AzInfected);
				m_AzInfected = null;
			}
			AzWorldLog("rest", 0, at);
			return;
		}
		if (m_AzStatesSlot >= AZ_STATES_SLOTS || m_AzWeatherClock < AZ_SEE_REST)
			return;
		m_AzWeatherClock = 0;
		vector fwd = player.GetDirection();
		fwd[1] = 0;
		fwd.Normalize();
		int side = m_AzStatesSlot - (m_AzStatesSlot / 5) * 5;
		float bearing = -25.0 + 12.5 * side;
		float dist = AzSeeDistance(m_AzStatesSlot);
		vector spot = AzOffset(at, fwd, bearing, dist);
		spot[1] = GetGame().SurfaceY(spot[0], spot[2]);
		string type = AzStatesType(m_AzStatesSlot);
		m_AzInfected = GetGame().CreateObject(type, spot, false, true, true);
		if (m_AzInfected)
			AzWorldLog("infected", dist, spot);
		else
			AzWorldLog("infected_failed", dist, spot);
		Print("[AssistZ] mode 34 slot " + m_AzStatesSlot.ToString() + " type " + type + " at " + dist.ToString() + " m bearing " + bearing.ToString());
		m_AzStatesSlot++;
		m_AzStatesOut = true;
	}

	// Mode 32: the character alone makes the noise. Invulnerable, with a loaded rifle in hands; Lab/harness/drive_own.py walks, runs, crawls and fires it and logs when. Anything that pings is the player's own.
	protected bool m_AzOwnArmed;

	protected void AzOwnSoundsTick(PlayerBase player, float dt)
	{
		if (m_AzBeaconWarmup < AZ_BEACON_WARMUP)
		{
			m_AzBeaconWarmup += dt;
			return;
		}
		player.SetAllowDamage(false);
		if (m_AzOwnArmed)
			return;
		m_AzOwnArmed = true;
		// A different rifle from run to run, so training on own shots is not training on one sound.
		string ownRifle = "AKM";
		int ownPick = Math.RandomInt(0, 4);
		if (ownPick == 1) ownRifle = "M4A1";
		if (ownPick == 2) ownRifle = "AugShort";
		if (ownPick == 3) ownRifle = "SVD";
		Weapon_Base rifle = Weapon_Base.Cast(player.GetHumanInventory().CreateInHands(ownRifle));
		if (rifle)
			rifle.SpawnAmmo("", WeaponWithAmmoFlags.CHAMBER);
		bool armed = rifle != null;
		Print("[AssistZ] mode 32 rifle " + ownRifle + " in hands ok=" + armed.ToString());
	}

	protected void AzInfectedLabTick(PlayerBase player, float dt)
	{
		if (m_AzBeaconWarmup < AZ_BEACON_WARMUP)
		{
			m_AzBeaconWarmup += dt;
			return;
		}
		player.SetAllowDamage(false);
		m_AzWeatherClock += dt;
		vector at = player.GetPosition();
		if (m_AzWeatherPhase == 0)
		{
			// Modes 18 and 21 repeat the attacks in heavy rain with no storm: mode 14's natural attack, in rain, read as thunder.
			if (AZ_LAB_MODE == 18 || AZ_LAB_MODE == 21)
				AzWeatherSet(1.0, 1.0, 0, 15);
			else
				AzWeatherSet(0.1, 0, 0, 15);
			AzWorldLog("clear", 0, at);
			m_AzWeatherPhase = 1;
			m_AzWeatherClock = 0;
			return;
		}
		if (m_AzWeatherPhase > 14)
			return;
		bool spawning = m_AzWeatherPhase <= 12 && m_AzWeatherPhase % 2 == 1;
		float wait = 20;
		if (m_AzWeatherPhase <= 12 && !spawning)
			wait = 45;
		if (m_AzWeatherPhase == 14)
			wait = 120;
		if (m_AzWeatherClock < wait)
			return;
		m_AzWeatherClock = 0;
		if (spawning)
		{
			vector fwd = player.GetDirection();
			fwd[1] = 0;
			fwd.Normalize();
			vector spot = AzOffset(at, fwd, 0, AZ_ATTACK_SPAWN_M);
			spot[1] = GetGame().SurfaceY(spot[0], spot[2]);
			string type = AzAttackType((m_AzWeatherPhase - 1) / 2);
			m_AzInfected = GetGame().CreateObject(type, spot, false, true, true);
			if (m_AzInfected)
				AzWorldLog("attack", AZ_ATTACK_SPAWN_M, spot);
			else
				AzWorldLog("attack_failed", AZ_ATTACK_SPAWN_M, spot);
			Print("[AssistZ] attack type " + type);
		}
		else if (m_AzWeatherPhase <= 12)
		{
			if (m_AzInfected)
			{
				GetGame().ObjectDelete(m_AzInfected);
				m_AzInfected = null;
			}
			AzWorldLog("rest", 0, at);
		}
		else if (m_AzWeatherPhase == 13)
		{
			AzWeatherSet(1.0, 1.0, 1.0, 15);
			AzWorldLog("storm", 15, at);
		}
		else
		{
			AzWeatherSet(0.1, 0, 0, 15);
			AzWorldLog("clear_after", 0, at);
		}
		m_AzWeatherPhase++;
	}

	// Modes 19 and 20: a survey of what the world sounds like when nothing is happening. A probe that called a repeating sound at one spot "sleigh crash" 74 times had only heard a few places. Clear weather, the invulnerable listener moved to each of 12 sites for AZ_SURVEY_SECONDS.
	// Two lists sharing no site: mode 19 trains, mode 20 never does. world.csv logs each arrival, the site number in the distance column.
	static const float AZ_SURVEY_SECONDS = 55.0;
	static const int AZ_SURVEY_SITES = 12;
	protected int m_AzSurveySite;

	protected vector AzSurveySpot(int index)
	{
		if (AZ_LAB_MODE == 22)
		{
			if (index == 0) return "10500 0 9800";
			if (index == 1) return "12900 0 8100";
			if (index == 2) return "12100 0 7200";
			if (index == 3) return "11300 0 6600";
			if (index == 4) return "9100 0 3900";
			if (index == 5) return "4400 0 4700";
			if (index == 6) return "5800 0 4800";
			if (index == 7) return "4500 0 6400";
			if (index == 8) return "4800 0 6800";
			if (index == 9) return "2700 0 9900";
			if (index == 10) return "6000 0 10300";
			return "8500 0 11900";
		}
		if (AZ_LAB_MODE == 23)
		{
			if (index == 0) return "5300 0 8600";
			if (index == 1) return "2200 0 11000";
			if (index == 2) return "2500 0 6300";
			if (index == 3) return "3300 0 4000";
			if (index == 4) return "1700 0 3800";
			if (index == 5) return "3400 0 4900";
			if (index == 6) return "9600 0 6500";
			if (index == 7) return "8400 0 6700";
			if (index == 8) return "7500 0 5200";
			if (index == 9) return "10100 0 5500";
			if (index == 10) return "12800 0 4400";
			return "7000 0 11000";
		}
		if (AZ_LAB_MODE == 19)
		{
			if (index == 0) return "6500 0 2500";
			if (index == 1) return "4500 0 2400";
			if (index == 2) return "2700 0 5300";
			if (index == 3) return "6100 0 7700";
			if (index == 4) return "3800 0 8900";
			if (index == 5) return "4500 0 10000";
			if (index == 6) return "9500 0 8800";
			if (index == 7) return "11900 0 12400";
			if (index == 8) return "12900 0 9700";
			if (index == 9) return "1700 0 14000";
			if (index == 10) return "3000 0 7800";
			return "7900 0 6500";
		}
		if (index == 0) return "10400 0 2200";
		if (index == 1) return "13400 0 6200";
		if (index == 2) return "1800 0 2200";
		if (index == 3) return "14000 0 13300";
		if (index == 4) return "10800 0 8100";
		if (index == 5) return "2000 0 7300";
		if (index == 6) return "8400 0 12700";
		if (index == 7) return "11600 0 14700";
		if (index == 8) return "12000 0 3500";
		if (index == 9) return "8000 0 3300";
		if (index == 10) return "11200 0 5500";
		return "1500 0 11900";
	}

	// Modes 26 and 27: labelled frames for the Vision Module (where am I, which way am I facing, from video alone). The listener stands at each point of a grid over the map, on land, looking in four directions, midday and clear. world.csv logs each view, its heading in the distance column; the frame grabber's file names give the time. Mode 26 is the index grid; mode 27 is offset by half a cell, so its points are places the index never saw.
	static const float AZ_MAP_ARRIVE_SECONDS = 8.0;    // the world streams in after a move
	static const float AZ_MAP_VIEW_SECONDS = 1.5;
	static const float AZ_MAP_START_X = AZ_MAP_START_X_VALUE;    // resume a survey from this column
	protected float m_AzMapX = -1;
	protected float m_AzMapZ;
	protected int m_AzMapView = -1;
	protected bool m_AzMapDone;

	protected bool AzMapNext()
	{
		float cell = 1000;
		float offset = 0;
		if (AZ_LAB_MODE == 27)
			offset = 500;
		float lowest = 500 + offset;
		while (true)
		{
			if (m_AzMapX < 0)
			{
				m_AzMapX = lowest;
				while (m_AzMapX < AZ_MAP_START_X)
					m_AzMapX += cell;
				m_AzMapZ = lowest;
			}
			else
			{
				m_AzMapZ += cell;
				if (m_AzMapZ > 15000)
				{
					m_AzMapZ = lowest;
					m_AzMapX += cell;
				}
			}
			if (m_AzMapX > 15000)
				return false;
			if (!GetGame().SurfaceIsSea(m_AzMapX, m_AzMapZ) && !GetGame().SurfaceIsPond(m_AzMapX, m_AzMapZ))
				return true;
		}
		return false;
	}

	protected void AzMapSurveyTick(PlayerBase player, float dt)
	{
		if (m_AzBeaconWarmup < AZ_BEACON_WARMUP)
		{
			m_AzBeaconWarmup += dt;
			return;
		}
		if (m_AzMapDone)
			return;
		player.SetAllowDamage(false);
		m_AzWeatherClock += dt;
		if (m_AzMapView < 0)
		{
			AzWeatherSet(0.1, 0, 0, 15);
			GetGame().GetWorld().SetDate(2026, 6, 15, 12, 0);
			if (!AzMapNext())
			{
				m_AzMapDone = true;
				AzWorldLog("done", 0, player.GetPosition());
				return;
			}
			vector spot = Vector(m_AzMapX, GetGame().SurfaceY(m_AzMapX, m_AzMapZ), m_AzMapZ);
			player.SetPosition(spot);
			m_AzMapView = 0;
			m_AzWeatherClock = 0;
			player.SetOrientation(Vector(0, 0, 0));
			return;
		}
		float wait = AZ_MAP_VIEW_SECONDS;
		if (m_AzMapView == 0)
			wait = AZ_MAP_ARRIVE_SECONDS;
		if (m_AzWeatherClock < wait)
			return;
		m_AzWeatherClock = 0;
		AzWorldLog("view", m_AzMapView * 90.0, player.GetPosition());
		m_AzMapView++;
		if (m_AzMapView > 3)
		{
			m_AzMapView = -1;
			return;
		}
		player.SetOrientation(Vector(m_AzMapView * 90.0, 0, 0));
	}

	protected void AzSurveyTick(PlayerBase player, float dt)
	{
		if (m_AzBeaconWarmup < AZ_BEACON_WARMUP)
		{
			m_AzBeaconWarmup += dt;
			return;
		}
		player.SetAllowDamage(false);
		m_AzWeatherClock += dt;
		if (m_AzSurveySite > 0 && m_AzWeatherClock < AZ_SURVEY_SECONDS)
			return;
		if (m_AzSurveySite >= AZ_SURVEY_SITES)
			return;
		if (m_AzSurveySite == 0)
			AzWeatherSet(0.1, 0, 0, 15);
		m_AzWeatherClock = 0;
		vector spot = AzSurveySpot(m_AzSurveySite);
		spot[1] = GetGame().SurfaceY(spot[0], spot[2]);
		player.SetPosition(spot);
		AzWorldLog("site", m_AzSurveySite, spot);
		m_AzSurveySite++;
	}

	protected void AzBeaconTick(PlayerBase player, float dt)
	{
		if (m_AzBeaconWarmup < AZ_BEACON_WARMUP)
		{
			m_AzBeaconWarmup += dt;
			return;
		}
		m_AzBeaconAccum += dt;
		if (m_AzBeaconAccum < AZ_BEACON_INTERVAL)
			return;
		m_AzBeaconAccum = 0;

		// (Enforce has no % operator.)
		if (m_AzBeaconIndex >= 16)
			m_AzBeaconIndex = 0;
		float bearing = AzBeaconBearing(m_AzBeaconIndex);
		float dist = AzBeaconDistance(m_AzBeaconIndex);
		m_AzBeaconIndex++;
		vector pos = player.GetPosition();
		vector fwd = player.GetDirection();
		fwd[1] = 0;
		fwd.Normalize();
		// DayZ: X = east, Z = north. Right of forward (fx, fz) is (fz, -fx); bearing is clockwise from forward.
		float rad = bearing * Math.DEG2RAD;
		float c = Math.Cos(rad);
		float sn = Math.Sin(rad);
		vector src = pos;
		src[0] = pos[0] + dist * (c * fwd[0] + sn * fwd[2]);
		src[2] = pos[2] + dist * (c * fwd[2] - sn * fwd[0]);
		src[1] = GetGame().SurfaceY(src[0], src[2]);

		Grenade_Base grenade = Grenade_Base.Cast(GetGame().CreateObjectEx("RGD5Grenade", src, ECE_PLACE_ON_SURFACE));
		if (!grenade)
		{
			Print("[AssistZ] beacon: could not create RGD5Grenade");
			return;
		}
		grenade.ActivateImmediate();

		float aimLR = 0;
		HumanCommandWeapons hcw = player.GetCommandModifier_Weapons();
		if (hcw)
			aimLR = hcw.GetBaseAimingAngleLR();
		vector ori = player.GetOrientation();
		float yaw = ori[0];
		int hour, minute, second;
		GetHourMinuteSecondUTC(hour, minute, second);
		string line = GetGame().GetTime().ToString() + "," + hour.ToStringLen(2) + ":" + minute.ToStringLen(2) + ":" + second.ToStringLen(2);
		line += "," + bearing.ToString() + "," + dist.ToString();
		line += "," + src[0].ToString() + "," + src[1].ToString() + "," + src[2].ToString();
		line += "," + pos[0].ToString() + "," + pos[1].ToString() + "," + pos[2].ToString();
		line += "," + yaw.ToString() + "," + aimLR.ToString();
		line += "," + fwd[0].ToString() + "," + fwd[2].ToString();
		FileHandle file = OpenFile(m_AzBeaconPath, FileMode.APPEND);
		if (file)
		{
			FPrintln(file, line);
			CloseFile(file);
		}
		Print("[AssistZ] beacon " + bearing.ToString() + " deg at " + dist.ToString() + " m");
	}

	// Aim change per command tick for segment i (radians; sign = direction). 0 = end of plan (stays still).
	protected float AzTurnRate(int i)
	{
		if (i == 0) return 0.004;
		if (i == 1) return -0.004;
		if (i == 2) return 0.01;
		if (i == 3) return -0.01;
		if (i == 4) return 0.025;
		if (i == 5) return -0.025;
		if (i == 6) return 0.007;
		if (i == 7) return -0.007;
		return 0;
	}

	protected void AzTurnLog(string what, float rate)
	{
		int hour, minute, second;
		GetHourMinuteSecondUTC(hour, minute, second);
		string line = GetGame().GetTime().ToString() + "," + hour.ToStringLen(2) + ":" + minute.ToStringLen(2) + ":" + second.ToStringLen(2);
		line += "," + what + "," + m_AzTurnIndex.ToString() + "," + rate.ToString();
		FileHandle file = OpenFile(m_AzTurnPath, FileMode.APPEND);
		if (file)
		{
			FPrintln(file, line);
			CloseFile(file);
		}
	}

	protected void AzTurnTick(PlayerBase player, float dt)
	{
		m_AzTurnClock += dt;
		HumanInputController input = player.GetInputController();
		if (!input)
			return;
		if (!m_AzTurning)
		{
			float wait = AZ_TURN_PAUSE;
			if (m_AzTurnIndex == 0)
				wait = AZ_TURN_WARMUP;
			if (m_AzTurnClock < wait)
				return;
			float rate = AzTurnRate(m_AzTurnIndex);
			if (rate == 0)
				return;
			input.OverrideAimChangeX(HumanInputControllerOverrideType.ENABLED, rate);
			m_AzTurning = true;
			m_AzTurnClock = 0;
			AzTurnLog("start", rate);
			return;
		}
		if (m_AzTurnClock < AZ_TURN_SECONDS)
			return;
		input.OverrideAimChangeX(HumanInputControllerOverrideType.DISABLED, 0);
		AzTurnLog("end", AzTurnRate(m_AzTurnIndex));
		m_AzTurning = false;
		m_AzTurnClock = 0;
		m_AzTurnIndex++;
	}

	protected PlayerBase AzFindPlayer(string name)
	{
		for (int i = 0; i < m_AzPlayers.Count(); i++)
		{
			PlayerBase p = PlayerBase.Cast(m_AzPlayers[i]);
			if (p && p.GetIdentity() && p.GetIdentity().GetName() == name)
				return p;
		}
		return null;
	}

	// A firing spot on land at dist from pos: straight ahead first, then bearings +-30, +-60 ... (no sea, no pond).
	protected vector AzLandSpot(vector pos, vector fwd, float dist)
	{
		vector spot = AzOffset(pos, fwd, 0, dist);
		for (int tryIndex = 1; tryIndex < 12; tryIndex++)
		{
			if (!GetGame().SurfaceIsSea(spot[0], spot[2]) && !GetGame().SurfaceIsPond(spot[0], spot[2]))
				break;
			float step = 30.0 * ((tryIndex + 1) / 2);
			if (tryIndex - (tryIndex / 2) * 2 == 0)
				step = -step;
			spot = AzOffset(pos, fwd, step, dist);
		}
		return spot;
	}

	// Mode 5: the real shooter. Per range: teleport the Shooter player to a land spot (invulnerable) and arm it with an
	// SVD + magazine; then per shot send it the aim point (beside the listener's head, near or wide). The Shooter's client
	// (lab mod AssistZLabShooter.c) aims and pulls the trigger itself.
	static const float AZ_REAL_INTERVAL = 6.0;
	protected float m_AzRealRange;

	protected void AzRealShooterTick(PlayerBase player, float dt)
	{
		PlayerBase shooter = AzFindPlayer("Shooter");
		if (!shooter)
			return;
		if (m_AzBeaconWarmup < AZ_BEACON_WARMUP)
		{
			m_AzBeaconWarmup += dt;
			return;
		}
		m_AzBeaconAccum += dt;
		if (m_AzBeaconAccum < AZ_REAL_INTERVAL)
			return;
		m_AzBeaconAccum = 0;
		float dist = AzSniperDistance(m_AzShotIndex);
		if (dist == 0)
			return;
		vector pos = player.GetPosition();
		vector fwd = player.GetDirection();
		fwd[1] = 0;
		fwd.Normalize();
		if (m_AzRealRange != dist)
		{
			vector spot = AzLandSpot(pos, fwd, dist);
			spot[1] = GetGame().SurfaceY(spot[0], spot[2]);
			shooter.SetPosition(spot);
			shooter.SetAllowDamage(false);
			player.SetAllowDamage(false);
			if (!shooter.GetHumanInventory().GetEntityInHands())
			{
				Weapon_Base rifle = Weapon_Base.Cast(shooter.GetHumanInventory().CreateInHands("SVD"));
				if (rifle)
					rifle.SpawnAmmo("", WeaponWithAmmoFlags.CHAMBER);
			}
			m_AzRealRange = dist;
			Print("[AssistZ] real shooter placed at " + dist.ToString() + " m");
			return;
		}
		float miss = AzSniperMiss(m_AzShotIndex);
		vector target = pos;
		target[0] = pos[0] + miss * fwd[2];
		target[2] = pos[2] - miss * fwd[0];
		target[1] = pos[1] + 1.7;
		Param1<vector> aimRpc = new Param1<vector>(target);
		shooter.RPCSingleParam(ASSISTZ_RPC_SHOOT, aimRpc, true, shooter.GetIdentity());

		vector src = shooter.GetPosition();
		float aimLR = 0;
		HumanCommandWeapons hcw = player.GetCommandModifier_Weapons();
		if (hcw)
			aimLR = hcw.GetBaseAimingAngleLR();
		vector ori = player.GetOrientation();
		float yaw = ori[0];
		int hour, minute, second;
		GetHourMinuteSecondUTC(hour, minute, second);
		string line = GetGame().GetTime().ToString() + "," + hour.ToStringLen(2) + ":" + minute.ToStringLen(2) + ":" + second.ToStringLen(2);
		line += "," + m_AzShotIndex.ToString() + "," + dist.ToString() + "," + miss.ToString() + ",sent";
		line += "," + src[0].ToString() + "," + src[1].ToString() + "," + src[2].ToString();
		line += "," + pos[0].ToString() + "," + pos[1].ToString() + "," + pos[2].ToString();
		line += "," + yaw.ToString() + "," + aimLR.ToString();
		FileHandle file = OpenFile(m_AzShotPath, FileMode.APPEND);
		if (file)
		{
			FPrintln(file, line);
			CloseFile(file);
		}
		Print("[AssistZ] real shot " + m_AzShotIndex.ToString() + " sent at " + dist.ToString() + " m miss " + miss.ToString());
		m_AzShotIndex++;
	}

	// Every update: keep the dummy's weapon RAISED, its body facing the aim point, and its weapon aim (HumanCommandWeapons
	// base aiming LR/UD, degrees) steered onto it. Lab 2026-11-03 (client barrel diagnostic): with a one-time raise and a
	// heading set only at fire time, the barrel on the client copy pointed 33-40 deg below the aim line (lowered carry
	// pose: the shots went into the ground, so no crack) and ~22 deg to the side (157 deg on a range's first shot).
	protected void AzShooterAimHold()
	{
		if (!m_AzShooter || !m_AzHasAimPoint)
			return;
		HumanInputController input = m_AzShooter.GetInputController();
		if (!input)
			return;
		input.OverrideRaise(HumanInputControllerOverrideType.ENABLED, true);
		vector eye = m_AzShooter.GetPosition();
		eye[1] = eye[1] + 1.6;
		vector toAim = m_AzAimPoint - eye;
		float facing = Math.Atan2(toAim[0], toAim[2]) * Math.RAD2DEG;
		vector ori = m_AzShooter.GetOrientation();
		ori[0] = facing;
		m_AzShooter.SetOrientation(ori);
		HumanCommandWeapons weapons = m_AzShooter.GetCommandModifier_Weapons();
		if (!weapons)
			return;
		float horizontal = Math.Sqrt(toAim[0] * toAim[0] + toAim[2] * toAim[2]);
		float wantPitch = Math.Atan2(toAim[1], horizontal) * Math.RAD2DEG;
		float pitchErr = wantPitch - weapons.GetBaseAimingAngleUD();
		float yawErr = 0 - weapons.GetBaseAimingAngleLR();
		input.OverrideAimChangeY(HumanInputControllerOverrideType.ONE_FRAME, Math.Clamp(pitchErr * 0.3, -6, 6) * Math.DEG2RAD);
		input.OverrideAimChangeX(HumanInputControllerOverrideType.ONE_FRAME, Math.Clamp(yawErr * 0.3, -6, 6) * Math.DEG2RAD);
	}

	// Mode 6: one Expansion AI Raider per range, AZ_EXP_SECONDS to engage, the listener invulnerable. The AI's own
	// behaviour raises, aims and fires (Expansion fires AI weapons on the client too, so the listener hears real shots).
	// 35 s per range (was 20: on 2026-11-03 run 4 the 100-400 m raiders were still closing in when their 20 s ended).
	static const float AZ_EXP_SECONDS = 35.0;
	protected PlayerBase m_AzAI;
	protected int m_AzExpRange;
	protected float m_AzExpClock;
	protected float m_AzExpDumpClock;
	protected float m_AzExpNoiseClock;
	// Mode 7 (wide misses): an East shooter 60 m ahead fires at an unarmed West decoy AZ_DECOY_OFFSET[range] m to the
	// listener's right; the listener is in a Civilian group (friendly to East), so the bullets aimed at the decoy pass him
	// at a chosen distance. TJ 2026-11-03: DayZ plays a close-only bullet noise - modes 6 only ever missed by < 5 m.
	protected PlayerBase m_AzDecoy;
	protected bool m_AzJoinedCivilian;
	protected string m_AzAIPosePath;
	protected float m_AzAIPoseClock;
	protected vector m_AzAILastPos;
	protected bool m_AzExpLooked;

	protected bool m_AzExpTickSeen;
	protected bool m_AzExpWarmSeen;
	protected bool m_AzSiteChosen;
	protected float m_AzSiteSettle;

	// Clear line of sight from eye height at a to eye height at b (a hit within 2 m of b counts as clear).
	protected bool AzLineClear(vector a, vector b)
	{
		vector from = a;
		from[1] = GetGame().SurfaceY(a[0], a[2]) + 1.6;
		vector to = b;
		to[1] = GetGame().SurfaceY(b[0], b[2]) + 1.6;
		vector hitPos;
		vector hitDir;
		int hitComponent;
		bool hit = DayZPhysics.RaycastRV(from, to, hitPos, hitDir, hitComponent, null, null, null, false, false, ObjIntersectView);
		return !hit || vector.Distance(hitPos, to) < 2.0;
	}

	// An open listening site (TJ, 2026-11-03: the random coastal spawns sit among bridges, boulders and railway
	// embankments). Candidates ring the three airfields' towers (positions from the mission's mapgrouppos.xml): 8
	// directions at 150 m and 300 m. Score = clear 400 m sight lines out of 8, sea/pond excluded; best one wins.
	protected vector AzPickOpenSite(out int bestScore)
	{
		array<vector> towers = new array<vector>;
		towers.Insert("4636.7 0 10445.3");   // North-West Airfield, ATC
		towers.Insert("4938.5 0 2471.9");    // Balota, control tower
		towers.Insert("11847.6 0 12463.7");  // Krasnostav, control tower
		vector best = towers[0];
		bestScore = -1;
		for (int t = 0; t < towers.Count(); t++)
		{
			for (int ring = 1; ring <= 2; ring++)
			{
				for (int k = 0; k < 8; k++)
				{
					float rad = k * 45.0 * Math.DEG2RAD;
					vector c = towers[t];
					c[0] = c[0] + 150.0 * ring * Math.Sin(rad);
					c[2] = c[2] + 150.0 * ring * Math.Cos(rad);
					if (GetGame().SurfaceIsSea(c[0], c[2]) || GetGame().SurfaceIsPond(c[0], c[2]))
						continue;
					int score = 0;
					for (int j = 0; j < 8; j++)
					{
						float r2 = j * 45.0 * Math.DEG2RAD;
						vector far = c;
						far[0] = c[0] + 400.0 * Math.Sin(r2);
						far[2] = c[2] + 400.0 * Math.Cos(r2);
						if (AzLineClear(c, far))
							score++;
					}
					if (score > bestScore)
					{
						bestScore = score;
						best = c;
					}
				}
			}
		}
		best[1] = GetGame().SurfaceY(best[0], best[2]);
		return best;
	}

	// A shooter spot on land with a clear line of sight from eye height to the listener's head: bearings 0, +-30, +-60 ...
	// (TJ, 2026-11-03: a building 15 m in front of the player - the AI cannot shoot what it cannot see).
	protected vector AzClearSpot(vector pos, vector fwd, float dist, vector listenerEye, out bool clear)
	{
		vector best = AzOffset(pos, fwd, 0, dist);
		clear = false;
		for (int tryIndex = 0; tryIndex < 12; tryIndex++)
		{
			float bearing = 30.0 * ((tryIndex + 1) / 2);
			if (tryIndex - (tryIndex / 2) * 2 == 0)
				bearing = -bearing;
			vector spot = AzOffset(pos, fwd, bearing, dist);
			if (GetGame().SurfaceIsSea(spot[0], spot[2]) || GetGame().SurfaceIsPond(spot[0], spot[2]))
				continue;
			vector eye = spot;
			eye[1] = GetGame().SurfaceY(spot[0], spot[2]) + 1.6;
			vector hitPos;
			vector hitDir;
			int hitComponent;
			bool hit = DayZPhysics.RaycastRV(eye, listenerEye, hitPos, hitDir, hitComponent, null, null, null, false, false, ObjIntersectView);
			if (!hit || vector.Distance(hitPos, listenerEye) < 1.5)
			{
				clear = true;
				return spot;
			}
		}
		return best;
	}

	protected void AzExpansionTick(PlayerBase player, float dt)
	{
		if (!m_AzExpTickSeen)
		{
			m_AzExpTickSeen = true;
			Print("[AssistZ] expansion tick reached");
		}
		if (m_AzBeaconWarmup < AZ_BEACON_WARMUP)
		{
			m_AzBeaconWarmup += dt;
			return;
		}
		if (!m_AzExpWarmSeen)
		{
			m_AzExpWarmSeen = true;
			Print("[AssistZ] expansion warmup done");
		}
		// Move the listener to an open site once, then let the client stream the area in before the first shooter.
		if (!m_AzSiteChosen)
		{
			m_AzSiteChosen = true;
			int siteScore;
			vector site = AzPickOpenSite(siteScore);
			player.SetPosition(site);
			Print("[AssistZ] listener moved to open site " + site.ToString() + " clear sight lines " + siteScore.ToString() + "/8");
			m_AzSiteSettle = 0;
			return;
		}
		if (m_AzSiteSettle < 8.0)
		{
			m_AzSiteSettle += dt;
			return;
		}
		player.SetAllowDamage(false);
		if ((AzUsesDecoy() || AZ_LAB_MODE == 10) && !m_AzJoinedCivilian)
		{
			m_AzJoinedCivilian = true;
			AssistZLabExpansion.JoinFaction(player, "Civilian");
			Print("[AssistZ] listener joined Civilian");
		}
		m_AzExpClock += dt;
		m_AzExpDumpClock += dt;
		if (m_AzAI && m_AzExpDumpClock >= 5.0)
		{
			m_AzExpDumpClock = 0;
			int hour;
			int minute;
			int year;
			int month;
			int day;
			GetGame().GetWorld().GetDate(year, month, day, hour, minute);
			string tag = "r" + m_AzExpRange.ToString() + " t" + m_AzExpClock.ToString() + " clock " + hour.ToString() + ":" + minute.ToString();
			AssistZLabExpansion.Dump(m_AzAI, player, tag);
		}
		// AI pose every 0.25 s: ground truth for what the AI did while the listener heard it (walking = footsteps).
		m_AzAIPoseClock += dt;
		if (m_AzAI && m_AzAIPoseClock >= 0.25)
		{
			vector aiPos = m_AzAI.GetPosition();
			float speed = vector.Distance(aiPos, m_AzAILastPos) / m_AzAIPoseClock;
			if (m_AzAILastPos == vector.Zero)
				speed = 0;
			m_AzAILastPos = aiPos;
			m_AzAIPoseClock = 0;
			float poseDist = vector.Distance(aiPos, player.GetPosition());
			string poseLine = GetGame().GetTime().ToString() + "," + m_AzExpRange.ToString() + "," + m_AzAI.GetType();
			poseLine += "," + aiPos[0].ToString() + "," + aiPos[1].ToString() + "," + aiPos[2].ToString();
			poseLine += "," + speed.ToString() + "," + poseDist.ToString() + "," + AssistZLabExpansion.FsmState(m_AzAI);
			FileHandle poseFile = OpenFile(m_AzAIPosePath, FileMode.APPEND);
			if (poseFile)
			{
				FPrintln(poseFile, poseLine);
				CloseFile(poseFile);
			}
		}
		// Every 3 s the listener makes a noise the raider can hear (radius 1.5 x its range); the look is pointed once.
		m_AzExpNoiseClock += dt;
		if (AZ_LAB_MODE != 10 && m_AzAI && m_AzExpNoiseClock >= 3.0 && AzNoiseOn(m_AzExpRange - 1))
		{
			m_AzExpNoiseClock = 0;
			PlayerBase noiseSource = player;
			if (AzUsesDecoy() && m_AzDecoy)
				noiseSource = m_AzDecoy;
			float aiDist = vector.Distance(m_AzAI.GetPosition(), noiseSource.GetPosition());
			float heard = AssistZLabExpansion.Provoke(m_AzAI, noiseSource, aiDist * 1.5, !m_AzExpLooked);
			if (!m_AzExpLooked)
				Print("[AssistZ] provoke r" + m_AzExpRange.ToString() + " noise radius " + heard.ToString() + " m at ai dist " + aiDist.ToString());
			m_AzExpLooked = true;
		}
		if (m_AzAI && m_AzExpClock < AZ_EXP_SECONDS)
			return;
		if (m_AzExpRange >= AzRangeCount())
		{
			if (m_AzAI)
				GetGame().ObjectDelete(m_AzAI);
			m_AzAI = null;
			if (m_AzDecoy)
				GetGame().ObjectDelete(m_AzDecoy);
			m_AzDecoy = null;
			return;
		}
		m_AzExpClock = 0;
		float dist = AzSniperDistance(m_AzExpRange * 4);
		m_AzExpRange++;
		if (m_AzAI)
			GetGame().ObjectDelete(m_AzAI);
		vector pos = player.GetPosition();
		vector fwd = player.GetDirection();
		fwd[1] = 0;
		fwd.Normalize();
		vector listenerEye = pos;
		listenerEye[1] = pos[1] + 1.6;
		bool clear;
		if (AzUsesDecoy())
		{
			// Decoy to the listener's right at the range's offset; the shooter ahead with a clear line to the decoy.
			if (m_AzDecoy)
				GetGame().ObjectDelete(m_AzDecoy);
			float offset = AzDecoyOffset(m_AzExpRange - 1);
			float slotBearing = 0;
			if (AZ_LAB_MODE == 25 || AZ_LAB_MODE == 28)
				slotBearing = (m_AzExpRange - 1) * 30.0;
			vector decoySpot = AzOffset(pos, fwd, slotBearing + 90, offset);
			vector decoyEye = decoySpot;
			decoyEye[1] = decoySpot[1] + 1.6;
			vector shooterSpot = AzClearSpot(pos, fwd, AzShooterDistance(m_AzExpRange - 1), decoyEye, clear);
			if (AZ_LAB_MODE == 25 || AZ_LAB_MODE == 28)
				shooterSpot = AzOffset(pos, fwd, slotBearing, AzShooterDistance(m_AzExpRange - 1));
			shooterSpot[1] = GetGame().SurfaceY(shooterSpot[0], shooterSpot[2]);
			// An armed decoy holds an EMPTY gun: it may raise the threat it poses, and it can add no shots to the log.
			m_AzDecoy = AssistZLabExpansion.SpawnAI(decoySpot, shooterSpot, "West", AzDecoyArmed(m_AzExpRange - 1), "AKM", false);
			string shooterWeapon = AzShooterWeapon(m_AzExpRange - 1);
			m_AzAI = AssistZLabExpansion.SpawnAI(shooterSpot, decoyEye, "East", true, shooterWeapon);
			Print("[AssistZ] mode " + AZ_LAB_MODE.ToString() + " decoy offset " + offset.ToString() + " m pos " + decoySpot.ToString() + " shooter pos " + shooterSpot.ToString() + " weapon " + shooterWeapon + " clear=" + clear.ToString());
			string decoyLine = GetGame().GetTime().ToString() + "," + m_AzExpRange.ToString() + ",decoy";
			decoyLine += "," + decoySpot[0].ToString() + "," + decoySpot[1].ToString() + "," + decoySpot[2].ToString() + ",0," + offset.ToString() + ",decoy";
			FileHandle decoyFile = OpenFile(m_AzAIPosePath, FileMode.APPEND);
			if (decoyFile)
			{
				FPrintln(decoyFile, decoyLine);
				CloseFile(decoyFile);
			}
		}
		else if (AZ_LAB_MODE == 10)
		{
			// Real footsteps, no gunfire: an unarmed AI walks a loop around the listener. The AI pose log says when it moved and how far
			// away. Each slot has its own ring (6 / 12 / 18 / 24 m) and speed limit (1 / 2 / 3 / 1).
			float walkRing = 6.0 * m_AzExpRange;
			int walkSpeed = m_AzExpRange;
			if (walkSpeed > 3) walkSpeed = 1;
			array<vector> walkLoop = new array<vector>;
			for (int walkPoint = 0; walkPoint < 6; walkPoint++)
			{
				vector loopSpot = AzOffset(pos, fwd, walkPoint * 60.0, walkRing);
				loopSpot[1] = GetGame().SurfaceY(loopSpot[0], loopSpot[2]);
				walkLoop.Insert(loopSpot);
			}
			m_AzAI = AssistZLabExpansion.SpawnWalker(walkLoop[0], listenerEye, walkLoop, walkSpeed);
			Print("[AssistZ] mode 10 walker ring " + walkRing.ToString() + " m speed limit " + walkSpeed.ToString());
		}
		else
		{
			vector spot = AzClearSpot(pos, fwd, dist, listenerEye, clear);
			spot[1] = GetGame().SurfaceY(spot[0], spot[2]);
			Print("[AssistZ] expansion spot at " + dist.ToString() + " m clear=" + clear.ToString());
			m_AzAI = AssistZLabExpansion.SpawnRaider(spot, listenerEye);
		}
		m_AzExpLooked = false;
		m_AzExpNoiseClock = 0;
		m_AzAILastPos = vector.Zero;
		if (!m_AzAI)
		{
			bool available = AssistZLabExpansion.Available();
			Print("[AssistZ] expansion AI spawn failed at " + dist.ToString() + " m (Expansion loaded=" + available.ToString() + ")");
			return;
		}
		vector at = m_AzAI.GetPosition();
		Print("[AssistZ] expansion AI raider at " + dist.ToString() + " m pos " + at.ToString());
	}

	// Mode 7 decoy offsets (m to the listener's right), one per range slot. Run 1 (2026-11-03, shooter 60 m: 5/15/30/60)
	// showed the precursor strong for <= 5 m passes from ~45 m shooters and weak for 9-44 m passes, but shooter distance
	// and pass distance changed together; run 2 (2/5/12/30) held the shooter at AZ_WIDE_SHOOTER_DIST and found the precursor
	// at every pass distance. Run 3 (1/2/25/30) gave a candidate flyby cue from only 7 close passes; the holdout runs (1/2/1/30)
	// collect more of them with far passes as controls.
	static const float AZ_WIDE_SHOOTER_DIST = 45.0;

	// Mode 8 (labelled set): every automatic shot so far was one accidental AUG burst at about 150 m, too little to train or test
	// a classifier on. Three weapons at three distances, slot i = weapon i / 3, distance i % 3. The decoy stands well aside so the
	// bullets pass far away, and the shot log names the weapon: the label comes from the game, not from reading the audio.
	protected bool AzUsesDecoy()
	{
		return AZ_LAB_MODE == 7 || AZ_LAB_MODE == 8 || AZ_LAB_MODE == 9 || AZ_LAB_MODE == 25 || AZ_LAB_MODE == 28;
	}

	// Mode 25: direction's first real test. AzClearSpot takes the first clear bearing nearest straight ahead, so every earlier shooter sat in a narrow arc in front. Here one AKM at 50 m per slot, 12 bearings 30 degrees apart all the way round, the decoy 30 m off to the shooter's side so bullets pass wide.
	protected int AzRangeCount()
	{
		if (AZ_LAB_MODE == 25) return 12;
		// Mode 28: distance test. Three weapons, each at 30, 55, 80 and 95 m, ranges the training never used, the bearing moving 30 degrees per slot.
		if (AZ_LAB_MODE == 28) return 12;
		if (AZ_LAB_MODE == 8) return 9;
		if (AZ_LAB_MODE == 9) return 6;
		return 4;
	}

	// Mode 9 asks where far shooters stop firing. First test, AKM (run sniper_20261103_235652), shots in 35 s: 90 m noise on and armed
	// decoy 54; noise on and unarmed 0-2 (mode 8); noise off 0, armed or not, so the noise is required and does not distract. 160 m: 0 in
	// every case. Second test (sniper_20261104_000348, another map site): noise on, armed decoy, 110 / 130 / 150 m, AKM then SVD: 0
	// shots in every slot (at 110 m active threat 0.032, not targeted, LOS true). So Expansion AI fires out to somewhere between 90 and
	// 110 m here. Why an Expansion patrol's AUG fired from 150 m in run 4 is not known.
	protected bool AzNoiseOn(int i)
	{
		return true;
	}

	protected bool AzDecoyArmed(int i)
	{
		return AZ_LAB_MODE == 8 || AZ_LAB_MODE == 9 || AZ_LAB_MODE == 25 || AZ_LAB_MODE == 28;
	}

	protected float AzShooterDistance(int i)
	{
		if (AZ_LAB_MODE == 25) return 50;
		if (AZ_LAB_MODE == 28)
		{
			int ladder = i - (i / 4) * 4;
			if (ladder == 0) return 30;
			if (ladder == 1) return 55;
			if (ladder == 2) return 80;
			return 95;
		}
		if (AZ_LAB_MODE == 9)
		{
			int far = i - (i / 3) * 3;
			if (far == 0) return 110;
			if (far == 1) return 130;
			return 150;
		}
		if (AZ_LAB_MODE != 8) return AZ_WIDE_SHOOTER_DIST;
		int slot = i - (i / 3) * 3;
		if (slot == 0) return 40;
		if (slot == 1) return 65;
		return 90;
	}

	protected string AzShooterWeapon(int i)
	{
		if (AZ_LAB_MODE == 29)
		{
			// Mode 29: 24 shots per weapon, 4 at each of 6 ranges (Expansion AI will not fire past about 100 m).
			int farFamily = i / 24;
			if (farFamily == 0) return "AKM";
			if (farFamily == 1) return "AugShort";
			if (farFamily == 2) return "M4A1";
			return "SVD";
		}
		if (AZ_LAB_MODE == 33)
		{
			// Weapons never in any training: two bolt rifles, a battle rifle, a submachine gun, two pistols. 16 shots each: 4 at each of 4 ranges.
			int newFamily = i / 16;
			if (newFamily == 0) return "Mosin9130";
			if (newFamily == 1) return "Winchester70";
			if (newFamily == 2) return "FAL";
			if (newFamily == 3) return "MP5K";
			if (newFamily == 4) return "CZ75";
			return "FNX45";
		}
		if (AZ_LAB_MODE == 25) return "AKM";
		if (AZ_LAB_MODE == 28)
		{
			int family = i / 4;
			if (family == 0) return "AKM";
			if (family == 1) return "AugShort";
			return "M4A1";
		}
		if (AZ_LAB_MODE == 9)
		{
			if (i < 3) return "AKM";
			return "SVD";
		}
		if (AZ_LAB_MODE != 8) return "SVD";
		int group = i / 3;
		if (group == 0) return "AugShort";
		if (group == 1) return "AKM";
		return "M4A1";
	}

	protected float AzDecoyOffset(int i)
	{
		if (AZ_LAB_MODE == 8 || AZ_LAB_MODE == 9 || AZ_LAB_MODE == 25 || AZ_LAB_MODE == 28) return 30;
		if (i <= 0) return 1;
		if (i == 1) return 2;
		if (i == 2) return 1;
		return 30;
	}

	protected float AzSniperDistance(int i)
	{
		if (AZ_LAB_MODE == 33)
		{
			if (i >= 96) return 0;
			int newStep = i / 4;
			newStep = newStep - (newStep / 4) * 4;
			if (newStep == 0) return 50;
			if (newStep == 1) return 150;
			if (newStep == 2) return 300;
			return 600;
		}
		if (AZ_LAB_MODE == 29)
		{
			if (i >= 96) return 0;
			int farStep = i / 4;
			farStep = farStep - (farStep / 6) * 6;
			if (farStep == 0) return 150;
			if (farStep == 1) return 200;
			if (farStep == 2) return 300;
			if (farStep == 3) return 400;
			if (farStep == 4) return 600;
			return 800;
		}
		if (i < 4) return 50;
		if (i < 8) return 100;
		if (i < 12) return 200;
		if (i < 16) return 400;
		return 0;
	}

	// Shots 0-1 of each range pass close, 2-3 wide. (Enforce has no % operator.)
	protected float AzSniperMiss(int i)
	{
		int inRange = i - (i / 4) * 4;
		if (inRange < 2) return AZ_MISS_OFFSET;
		return AZ_WIDE_OFFSET;
	}

	protected void AzSniperTick(PlayerBase player, float dt)
	{
		if (m_AzBeaconWarmup < AZ_BEACON_WARMUP)
		{
			m_AzBeaconWarmup += dt;
			return;
		}
		m_AzBeaconAccum += dt;
		if (m_AzBeaconAccum < AZ_SNIPER_INTERVAL)
			return;
		m_AzBeaconAccum = 0;
		float dist = AzSniperDistance(m_AzShotIndex);
		if (dist == 0)
			return;

		vector pos = player.GetPosition();
		vector fwd = player.GetDirection();
		fwd[1] = 0;
		fwd.Normalize();
		// The shooter stands on land: from straight ahead, try bearings +-30, +-60 ... until neither sea nor pond (TJ,
		// 2026-11-03: the player spawned facing the sea and the shooter spawned in the water).
		// Mode 29 moves round the listener with each range: 24 ranges at 75 degree steps land on 24 different bearings.
		float baseBearing = 0;
		if (AZ_LAB_MODE == 29 || AZ_LAB_MODE == 33)
			baseBearing = 75.0 * (m_AzShotIndex / 4);
		float shooterBearing = baseBearing;
		vector src = AzOffset(pos, fwd, baseBearing, dist);
		for (int tryIndex = 1; tryIndex < 12; tryIndex++)
		{
			if (!GetGame().SurfaceIsSea(src[0], src[2]) && !GetGame().SurfaceIsPond(src[0], src[2]))
				break;
			float step = 30.0 * ((tryIndex + 1) / 2);
			if (tryIndex - (tryIndex / 2) * 2 == 0)
				step = -step;
			shooterBearing = baseBearing + step;
			src = AzOffset(pos, fwd, shooterBearing, dist);
		}
		src[1] = src[1] + 1.5;
		// Aim point: the miss distance to the player's right, at head height.
		float miss = AzSniperMiss(m_AzShotIndex);
		vector target = pos;
		target[0] = pos[0] + miss * fwd[2];
		target[2] = pos[2] - miss * fwd[0];
		target[1] = pos[1] + 1.7;
		m_AzAimPoint = target;
		m_AzHasAimPoint = true;

		// Stage 1 (new range): spawn the shooter there, invulnerable. Stage 2: put the SVD in its hands. Then fire.
		if (!m_AzShooter || m_AzShooterRange != dist)
		{
			if (m_AzShooter)
				GetGame().ObjectDelete(m_AzShooter);
			m_AzShooter = PlayerBase.Cast(GetGame().CreateObjectEx("SurvivorM_Mirek", src, ECE_PLACE_ON_SURFACE | ECE_INITAI | ECE_EQUIP_ATTACHMENTS));
			m_AzShooterRange = dist;
			m_AzArmed = false;
			if (m_AzShooter)
			{
				m_AzShooter.OnSpawnedFromConsole();
				m_AzShooter.SetAllowDamage(false);
			}
			bool spawned = m_AzShooter != null;
			Print("[AssistZ] shooter spawned at " + dist.ToString() + " m ok=" + spawned.ToString());
			return;
		}
		Weapon_Base rifle = Weapon_Base.Cast(m_AzShooter.GetHumanInventory().GetEntityInHands());
		if (!m_AzArmed || !rifle)
		{
			if (rifle)
				GetGame().ObjectDelete(rifle);
			rifle = Weapon_Base.Cast(m_AzShooter.GetHumanInventory().CreateInHands(AzShooterWeapon(m_AzShotIndex)));
			bool armed = rifle != null;
			if (rifle)
			{
				// A full magazine plus a chambered round, as a player would carry it.
				rifle.SpawnAmmo("", WeaponWithAmmoFlags.CHAMBER);
				// Weapon raised (a player holds aim to shoot).
				m_AzShooter.GetInputController().OverrideRaise(HumanInputControllerOverrideType.ENABLED, true);
			}
			m_AzArmed = armed;
			Print("[AssistZ] shooter fresh rifle in hands ok=" + armed.ToString());
			return;
		}
		m_AzArmed = false;
		// Face the aim point (so the shooter's pose matches the shot), then have the client fire (below).
		vector muzzle = m_AzShooter.GetPosition();
		muzzle[1] = muzzle[1] + 1.5;
		vector aim = target - muzzle;
		aim.Normalize();
		float facing = Math.Atan2(aim[0], aim[2]) * Math.RAD2DEG;
		vector shooterOri = m_AzShooter.GetOrientation();
		shooterOri[0] = facing;
		m_AzShooter.SetOrientation(shooterOri);
		// Pitch is held continuously by AzShooterAimHold (a single one-frame nudge at fire time left the shots low: TJ saw
		// them hit the ground).
		bool loaded = !rifle.IsChamberEmpty(0);
		// The shot is fired on the listening client's copy of the shooter (ASSISTZ_RPC_FIRE in the lab mod): the client
		// engine simulates the projectile, so report, crack, flyby and their timing are the game's own. Server-side firing
		// (Weapon.Fire / WeaponManager.Fire / PostWeaponEvent) produced impacts on the client but no sound.
		Param2<vector, vector> fireRpc = new Param2<vector, vector>(muzzle, aim);
		m_AzShooter.RPCSingleParam(ASSISTZ_RPC_FIRE, fireRpc, true, player.GetIdentity());
		bool fired = loaded;
		src = muzzle;

		string shotWeapon = rifle.GetType();
		Print("[AssistZ] sniper shot " + m_AzShotIndex.ToString() + " at " + dist.ToString() + " m fired=" + fired.ToString() + " weapon=" + shotWeapon);

		float aimLR = 0;
		HumanCommandWeapons hcw = player.GetCommandModifier_Weapons();
		if (hcw)
			aimLR = hcw.GetBaseAimingAngleLR();
		vector ori = player.GetOrientation();
		float yaw = ori[0];
		int hour, minute, second;
		GetHourMinuteSecondUTC(hour, minute, second);
		string line = GetGame().GetTime().ToString() + "," + hour.ToStringLen(2) + ":" + minute.ToStringLen(2) + ":" + second.ToStringLen(2);
		line += "," + m_AzShotIndex.ToString() + "," + dist.ToString() + "," + miss.ToString() + "," + fired.ToString();
		line += "," + src[0].ToString() + "," + src[1].ToString() + "," + src[2].ToString();
		line += "," + pos[0].ToString() + "," + pos[1].ToString() + "," + pos[2].ToString();
		line += "," + yaw.ToString() + "," + aimLR.ToString() + "," + shotWeapon;
		FileHandle file = OpenFile(m_AzShotPath, FileMode.APPEND);
		if (file)
		{
			FPrintln(file, line);
			CloseFile(file);
		}
		m_AzShotIndex++;
	}

	protected vector AzOffset(vector pos, vector fwd, float bearing, float dist)
	{
		float rad = bearing * Math.DEG2RAD;
		float c = Math.Cos(rad);
		float sn = Math.Sin(rad);
		vector src = pos;
		src[0] = pos[0] + dist * (c * fwd[0] + sn * fwd[2]);
		src[2] = pos[2] + dist * (c * fwd[2] - sn * fwd[0]);
		src[1] = GetGame().SurfaceY(src[0], src[2]);
		return src;
	}

	protected void AzFixedTick(PlayerBase player, float dt)
	{
		if (m_AzBeaconWarmup < AZ_BEACON_WARMUP)
		{
			m_AzBeaconWarmup += dt;
			return;
		}
		m_AzBeaconAccum += dt;
		if (m_AzBeaconAccum < AZ_FIXED_INTERVAL)
			return;
		m_AzBeaconAccum = 0;
		vector pos = player.GetPosition();
		vector fwd = player.GetDirection();
		fwd[1] = 0;
		fwd.Normalize();
		if (!m_AzFixedSet)
		{
			m_AzFixedA = AzOffset(pos, fwd, 40, AZ_FIXED_DISTANCE);
			m_AzFixedB = AzOffset(pos, fwd, 250, AZ_FIXED_DISTANCE);
			m_AzFixedSet = true;
		}
		vector src = m_AzFixedB;
		if (m_AzFixedNextIsA)
			src = m_AzFixedA;
		m_AzFixedNextIsA = !m_AzFixedNextIsA;

		Grenade_Base grenade = Grenade_Base.Cast(GetGame().CreateObjectEx("RGD5Grenade", src, ECE_PLACE_ON_SURFACE));
		if (!grenade)
			return;
		grenade.ActivateImmediate();

		// Bearing of the source relative to the body's forward right now (clockwise, degrees).
		float dx = src[0] - pos[0];
		float dz = src[2] - pos[2];
		float along = dx * fwd[0] + dz * fwd[2];
		float right = dx * fwd[2] - dz * fwd[0];
		float bearing = Math.Atan2(right, along) * Math.RAD2DEG;
		float dist = Math.Sqrt(dx * dx + dz * dz);
		float aimLR = 0;
		HumanCommandWeapons hcw = player.GetCommandModifier_Weapons();
		if (hcw)
			aimLR = hcw.GetBaseAimingAngleLR();
		vector ori = player.GetOrientation();
		float yaw = ori[0];
		int hour, minute, second;
		GetHourMinuteSecondUTC(hour, minute, second);
		string line = GetGame().GetTime().ToString() + "," + hour.ToStringLen(2) + ":" + minute.ToStringLen(2) + ":" + second.ToStringLen(2);
		line += "," + bearing.ToString() + "," + dist.ToString();
		line += "," + src[0].ToString() + "," + src[1].ToString() + "," + src[2].ToString();
		line += "," + pos[0].ToString() + "," + pos[1].ToString() + "," + pos[2].ToString();
		line += "," + yaw.ToString() + "," + aimLR.ToString();
		line += "," + fwd[0].ToString() + "," + fwd[2].ToString();
		FileHandle file = OpenFile(m_AzBeaconPath, FileMode.APPEND);
		if (file)
		{
			FPrintln(file, line);
			CloseFile(file);
		}
	}

	protected float AzBeaconBearing(int i)
	{
		if (i == 0) return 45;
		if (i == 1) return 135;
		if (i == 2) return 315;
		if (i == 3) return 225;
		if (i == 4) return 90;
		if (i == 5) return 270;
		if (i == 6) return 0;
		if (i == 7) return 180;
		if (i < 12) return 90;
		return 0;
	}

	protected float AzBeaconDistance(int i)
	{
		if (i < 8) return 50;
		if (i == 8 || i == 12) return 35;
		if (i == 9 || i == 13) return 70;
		if (i == 10 || i == 14) return 140;
		return 280;
	}

	protected void AzPoseFlush()
	{
		if (!m_AzPoseOpen || m_AzPoseRows.Count() == 0)
			return;
		FileHandle file = OpenFile(m_AzPosePath, FileMode.APPEND);
		if (!file)
			return;
		for (int i = 0; i < m_AzPoseRows.Count(); i++)
			FPrintln(file, m_AzPoseRows[i]);
		CloseFile(file);
		m_AzPoseRows.Clear();
	}

	override void OnMissionFinish()
	{
		AzPoseFlush();
		m_AzPoseOpen = false;
		super.OnMissionFinish();
	}
	// ---- end AssistZ lab -------------------------------------------------------------------------------------

