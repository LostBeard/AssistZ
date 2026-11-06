// AssistZ lab: the DayZ-Expansion-AI calls live HERE, in the lab mod, because the mission's init.c is compiled without the
// mods' preprocessor defines (2026-11-03: '#ifdef EXPANSIONMODAI' in init.c was silently compiled out even though the
// World and Mission modules both define it). Without Expansion loaded this compiles to stubs, so the mod still loads.
// Expansion (CC BY-NC-ND 4.0) is only used through its API; none of its files are modified or repacked.
class AssistZLabExpansion
{
	static bool Available()
	{
#ifdef EXPANSIONMODAI
		return true;
#else
		return false;
#endif
	}

	// An Expansion AI survivor in a hostile Raiders group with an SVD + magazine at spot (invulnerable, so a run's
	// ranges are not cut short). Its own behaviour raises, aims and fires at hostiles it can see.
	// Spawned FACING lookAt (2026-11-03 run 2: each raider listed the listener as an enemy, absolute threat 1.3, but
	// "LOS state false" - spawned facing a random way, the listener was outside its field of view (+-99 deg at 50 m,
	// narrower further out), so the active threat stayed 0 and it idled without ever turning).
	static PlayerBase SpawnRaider(vector spot, vector lookAt)
	{
		return SpawnAI(spot, lookAt, "Raiders", true);
	}

	// An Expansion AI survivor of a faction ("Raiders", "East", "West", ...) facing lookAt, invulnerable, with a weapon + a
	// magazine when armed (the decoy of lab modes 7 and 8 is unarmed: a target, not a shooter).
	static PlayerBase SpawnAI(vector spot, vector lookAt, string faction, bool armed, string weapon = "SVD", bool loaded = true)
	{
#ifdef EXPANSIONMODAI
		eAIBase ai = eAIBase.Cast(GetGame().CreateObject(eAISurvivor.GetQuasiRandom(), spot));
		if (!ai)
			return null;
		vector face = lookAt - spot;
		face[1] = 0;
		face.Normalize();
		ai.SetDirection(face);
		ai.SetOrientation(face.VectorToAngles());
		eAIGroup group = eAIGroup.CreateGroup(eAIFaction.Create(faction));
		ai.SetGroup(group);
		if (armed)
		{
			Weapon_Base rifle = Weapon_Base.Cast(ai.GetHumanInventory().CreateInHands(weapon));
			if (rifle && loaded)
				rifle.SpawnAmmo("", WeaponWithAmmoFlags.CHAMBER);
		}
		ai.SetAllowDamage(false);
		return ai;
#else
		return null;
#endif
	}

	// The listener makes a noise at his own position, through Expansion's own noise system (the path the game uses for a
	// player's action noises), so the raider hears him, turns, and its own targeting finds him. Run 3 (2026-11-03): with no
	// stimulus the raiders idled looking elsewhere (look 38 deg vs bearing 165 deg) and Expansion only line-of-sight checks
	// the CURRENT target, against the look direction. Also points the raider's look at the listener's head once.
	// radius = the wanted audible radius (m); Expansion's audible radius is the config strength x the multiplier. Returns the
	// radius actually used (or -1 if the noise config has no strength).
	static float Provoke(PlayerBase ai, PlayerBase listener, float radius, bool look)
	{
#ifdef EXPANSIONMODAI
		eAIBase e = eAIBase.Cast(ai);
		if (!e || !listener)
			return 0;
		string path = "CfgVehicles SurvivorBase NoiseActionDefault";
		eAINoiseParams params = eAINoiseSystem.GetNoiseParams(path);
		if (!params || params.m_Strength <= 0)
			return -1;
		float multiplier = radius / params.m_Strength;
		eAINoiseSystem.AddNoise(listener, listener.GetPosition(), path, multiplier);
		if (look)
		{
			vector head = listener.GetPosition();
			head[1] = head[1] + 1.6;
			e.LookAtPosition(head);
		}
		return params.m_Strength * multiplier;
#else
		return 0;
#endif
	}

	// Lab mode 10: an unarmed AI walking a waypoint loop at a given speed limit. Noises do not do this: an unarmed friendly AI stayed
	// Idle on all 467 pose samples of run sniper_20261104_005406.
	static PlayerBase SpawnWalker(vector spot, vector lookAt, array<vector> loop, int speedLimit)
	{
#ifdef EXPANSIONMODAI
		eAIBase ai = eAIBase.Cast(SpawnAI(spot, lookAt, "East", false));
		if (!ai)
			return null;
		eAIGroup group = ai.GetGroup();
		foreach (vector point : loop)
			group.AddWaypoint(point);
		group.SetWaypointBehaviour(eAIWaypointBehavior.LOOP);
		ai.SetMovementSpeedLimit(speedLimit);
		return ai;
#else
		return null;
#endif
	}

	// Puts a (real) player in an Expansion faction group. Lab mode 7: the listener joins "Civilian", which "East" treats
	// as friendly (eAIFactionEast.IsFriendly), so the East shooter fires at the West decoy beside him, not at him.
	static void JoinFaction(PlayerBase player, string faction)
	{
#ifdef EXPANSIONMODAI
		if (player)
			player.SetGroup(eAIGroup.CreateGroup(eAIFaction.Create(faction)));
#endif
	}

	// The AI's Expansion FSM state, "State>SubState" (e.g. "Fighting>FireWeapon"), for the AI pose log.
	static string FsmState(PlayerBase ai)
	{
#ifdef EXPANSIONMODAI
		eAIBase e = eAIBase.Cast(ai);
		if (!e)
			return "";
		auto fsm = e.GetFSM();
		if (!fsm)
			return "none";
		auto state = fsm.GetState();
		if (!state)
			return "none";
		string name = state.GetName();
		auto sub = state.GetSubFSM();
		if (sub)
		{
			auto subState = sub.GetState();
			if (subState)
				name = name + ">" + subState.GetName();
		}
		return name;
#else
		return "";
#endif
	}

	// What the AI itself thinks of the listener (TJ, 2026-11-03: the raiders never aimed, shot or even looked his way):
	// Expansion's own enemy test, its cached threat for the listener, then its full state report (threat to self, look and
	// aim angles, tracked targets, FSM). Read from the game, not inferred.
	static void Dump(PlayerBase ai, PlayerBase listener, string tag)
	{
#ifdef EXPANSIONMODAI
		eAIBase e = eAIBase.Cast(ai);
		if (!e || !listener)
			return;
		bool moving;
		bool friendly;
		bool targeted;
		bool enemy = e.PlayerIsEnemy(listener, false, moving, friendly, targeted);
		float threat = e.eAI_GetCachedThreat(listener.GetTargetInformation());
		float dist = vector.Distance(e.GetPosition(), listener.GetPosition());
		string hands = "none";
		EntityAI inHands = listener.GetHumanInventory().GetEntityInHands();
		if (inHands)
			hands = inHands.GetType();
		vector toListener = listener.GetPosition() - e.GetPosition();
		vector toAngles = toListener.VectorToAngles();
		float bearing = toAngles[0];
		Print("[AssistZ] eai " + tag + " bearingToListener " + bearing.ToString() + " dist " + dist.ToString() + " enemy " + enemy.ToString() + " friendly " + friendly.ToString() + " targeted " + targeted.ToString() + " threatOfListener " + threat.ToString() + " listenerHands " + hands);
		TStringArray report = e.eAI_DumpState(null, false, "");
		foreach (string line : report)
			Print("[AssistZ] eai " + tag + " " + line);
#endif
	}
};
