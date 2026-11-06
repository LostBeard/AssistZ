// AssistZ lab: make a server-controlled character's gunshot real on the listening client, with NO modelled timing.
//
// The vanilla engine does not replicate a server-controlled character's shot to clients (lab, 2026-11-03: impacts arrive,
// the report, crack and flyby do not - through Weapon.Fire, WeaponManager.Fire and PostWeaponEvent alike).
// DayZ-Expansion-AI solves the same problem by firing AI weapons on the client outside the vanilla weapon FSM. The lab
// does the same: the server sends ASSISTZ_RPC_FIRE to the shooter ENTITY itself, and the client calls Fire on the rifle in
// that character's hands. The client's own engine then simulates the projectile, so the report, the supersonic crack,
// the flyby and their timing all come from the game, not from this script.
//
// (An earlier version played the crack from a physics model - distance / 830 m/s - which only reproduced the model:
// circular, removed. TJ: "are you assuming how the game does the sounds?")
const int ASSISTZ_RPC_FIRE = 0x415A4649; // "AZFI"

modded class PlayerBase
{
	override void OnRPC(PlayerIdentity sender, int rpc_type, ParamsReadContext ctx)
	{
		super.OnRPC(sender, rpc_type, ctx);
		if (rpc_type != ASSISTZ_RPC_FIRE)
			return;
		Param2<vector, vector> shot = new Param2<vector, vector>("0 0 0", "0 0 0");
		if (!ctx.Read(shot))
			return;
		if (GetGame().IsDedicatedServer())
			return;
		Weapon_Base rifle = Weapon_Base.Cast(GetHumanInventory().GetEntityInHands());
		if (!rifle)
		{
			Print("[AssistZLab] client fire: no weapon in hands on " + GetType());
			return;
		}
		// The game's own trigger path, run on this client's copy of the shooter: the weapon state machine plays the fire
		// animation, whose events play the weapon's configured report (soundSetShot / soundSetShotExt: the engine picks
		// interior, suppressor, improvised suppressor), and fires the projectile (crack, flyby). Weapon.Fire alone (the
		// previous version) gave the engine's projectile sounds but no report.
		if (rifle.IsChamberEmpty(0))
		{
			string round = "Bullet_762x54";
			if (rifle.GetType() == "AKM")
				round = "Bullet_762x39";
			if (rifle.GetType() == "AugShort" || rifle.GetType() == "M4A1")
				round = "Bullet_556x45";
			if (rifle.GetType() == "Winchester70" || rifle.GetType() == "FAL")
				round = "Bullet_308Win";
			if (rifle.GetType() == "MP5K" || rifle.GetType() == "CZ75")
				round = "Bullet_9x19";
			if (rifle.GetType() == "FNX45")
				round = "Bullet_45ACP";
			rifle.PushCartridgeToChamber(0, 1, round);
		}
		// Diagnostic: where the barrel really points on this client (memory points, as the game's own barrel-length code
		// reads them) against the intended direction from the server, as yaw/pitch errors in degrees.
		vector muzzleEnd = rifle.ModelToWorld(rifle.GetSelectionPositionLS("usti hlavne"));
		vector breechEnd = rifle.ModelToWorld(rifle.GetSelectionPositionLS("konec hlavne"));
		vector barrel = muzzleEnd - breechEnd;
		barrel.Normalize();
		vector wanted = shot.param2;
		float yawErr = (Math.Atan2(barrel[0], barrel[2]) - Math.Atan2(wanted[0], wanted[2])) * Math.RAD2DEG;
		if (yawErr > 180)
			yawErr = yawErr - 360;
		if (yawErr < -180)
			yawErr = yawErr + 360;
		float barrelPitch = Math.Asin(barrel[1]) * Math.RAD2DEG;
		float wantedPitch = Math.Asin(wanted[1]) * Math.RAD2DEG;
		float pitchErr = barrelPitch - wantedPitch;
		GetWeaponManager().Fire(rifle);
		Print("[AssistZLab] client trigger " + rifle.GetType() + " at " + GetPosition().ToString() + " barrelYawErr=" + yawErr.ToString() + " barrelPitchErr=" + pitchErr.ToString());
	}
};
