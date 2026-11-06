// AssistZ lab: a REAL player shooter. The lab runs a second DayZDiag client ("Shooter", -nosound) whose local player is
// driven by this script: the server sends ASSISTZ_RPC_SHOOT with an aim point; this client steers its own camera onto
// that point with the input overrides, raises the rifle and pulls the trigger through the normal local-player path
// (WeaponManager.Fire). The game then replicates the shot to the listening client exactly as any player's shot: report,
// crack, flyby, suppressor/interior sound selection and their timing are all the engine's, nothing is modelled here.
// (Server-controlled dummies were tried first: their shots never reached clients as sound - see AssistZLabShots.c.)
const int ASSISTZ_RPC_SHOOT = 0x415A5354; // "AZST"

modded class PlayerBase
{
	protected vector m_AzAimTarget;
	protected bool m_AzAiming;
	protected int m_AzAimTicks;
	protected int m_AzSteadyTicks;

	override void OnRPC(PlayerIdentity sender, int rpc_type, ParamsReadContext ctx)
	{
		super.OnRPC(sender, rpc_type, ctx);
		if (rpc_type != ASSISTZ_RPC_SHOOT)
			return;
		Param1<vector> aim = new Param1<vector>("0 0 0");
		if (!ctx.Read(aim) || GetGame().IsDedicatedServer())
			return;
		// Only the shooter's own client steers its own player. (Comparing this with GetGame().GetPlayer() directly is an
		// "unsafe down-casting" compile error in Enforce: Man vs PlayerBase.)
		PlayerBase localPlayer = PlayerBase.Cast(GetGame().GetPlayer());
		if (localPlayer != this)
			return;
		m_AzAimTarget = aim.param1;
		m_AzAimTicks = 0;
		m_AzSteadyTicks = 0;
		if (!m_AzAiming)
		{
			m_AzAiming = true;
			GetGame().GetCallQueue(CALL_CATEGORY_GAMEPLAY).CallLater(AzAimTick, 16, true);
		}
		GetInputController().OverrideRaise(HumanInputControllerOverrideType.ENABLED, true);
	}

	// Heading in degrees clockwise from +Z (north), pitch in degrees up, of a direction vector.
	protected float AzHeading(vector d) { return Math.Atan2(d[0], d[2]) * Math.RAD2DEG; }
	protected float AzPitch(vector d) { return Math.Atan2(d[1], Math.Sqrt(d[0] * d[0] + d[2] * d[2])) * Math.RAD2DEG; }

	void AzAimTick()
	{
		if (!m_AzAiming)
			return;
		m_AzAimTicks++;
		vector cam = GetGame().GetCurrentCameraPosition();
		vector want = m_AzAimTarget - cam;
		vector have = GetGame().GetCurrentCameraDirection();
		float yawErr = AzHeading(want) - AzHeading(have);
		if (yawErr > 180) yawErr = yawErr - 360;
		if (yawErr < -180) yawErr = yawErr + 360;
		float pitchErr = AzPitch(want) - AzPitch(have);
		// Proportional steering, a fraction of the error per tick (aim change is in radians per command step).
		GetInputController().OverrideAimChangeX(HumanInputControllerOverrideType.ONE_FRAME, Math.Clamp(yawErr * 0.25, -8, 8) * Math.DEG2RAD);
		GetInputController().OverrideAimChangeY(HumanInputControllerOverrideType.ONE_FRAME, Math.Clamp(pitchErr * 0.25, -8, 8) * Math.DEG2RAD);
		if (Math.AbsFloat(yawErr) < 0.3 && Math.AbsFloat(pitchErr) < 0.3)
			m_AzSteadyTicks++;
		else
			m_AzSteadyTicks = 0;
		// Fire once the aim has held for ~0.25 s (the raise animation has finished by then), or give up after ~4 s.
		if (m_AzSteadyTicks >= 15 || m_AzAimTicks > 250)
		{
			Weapon_Base rifle = Weapon_Base.Cast(GetHumanInventory().GetEntityInHands());
			bool armed = false;
			if (rifle)
			{
				armed = true;
				GetWeaponManager().Fire(rifle);
			}
			Print("[AssistZLab] shooter fired=" + armed.ToString() + " yawErr=" + yawErr.ToString() + " pitchErr=" + pitchErr.ToString() + " ticks=" + m_AzAimTicks.ToString());
			m_AzAiming = false;
			GetGame().GetCallQueue(CALL_CATEGORY_GAMEPLAY).Remove(AzAimTick);
		}
	}
};
