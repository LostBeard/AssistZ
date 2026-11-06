// AssistZ lab: ground truth for every shot (lab mode 6: DayZ-Expansion-AI shooters), logged on whichever side fires it -
// Expansion fires AI weapons on the client too (outside the vanilla weapon FSM), and the first mode-6 run logged nothing
// on the server. Each shot's game time, side, shooter, muzzle position and barrel direction (the weapon's own memory
// points, as the game's barrel-length code reads them) are appended to $profile:assistz_aishots.csv of that process. The analysis computes the bullet line's true closest approach
// to the listener's head from this - nothing about the sound is modelled here.
modded class Weapon_Base
{
	override void EEFired(int muzzleType, int mode, string ammoType)
	{
		super.EEFired(muzzleType, mode, ammoType);
		string side = "client";
		if (GetGame().IsServer())
			side = "server";
		vector muzzle = ModelToWorld(GetSelectionPositionLS("usti hlavne"));
		vector breech = ModelToWorld(GetSelectionPositionLS("konec hlavne"));
		vector barrel = muzzle - breech;
		barrel.Normalize();
		string shooter = "none";
		Man owner = GetHierarchyRootPlayer();
		if (owner)
			shooter = owner.GetType();
		string line = GetGame().GetTime().ToString() + "," + side + "," + GetType() + "," + shooter + "," + ammoType;
		line += "," + muzzle[0].ToString() + "," + muzzle[1].ToString() + "," + muzzle[2].ToString();
		line += "," + barrel[0].ToString() + "," + barrel[1].ToString() + "," + barrel[2].ToString();
		string path = "$profile:assistz_aishots.csv";
		bool exists = FileExist(path);
		FileHandle file = OpenFile(path, FileMode.APPEND);
		if (!file)
			return;
		if (!exists)
			FPrintln(file, "game_ms,side,weapon,shooter,ammo,mx,my,mz,dx,dy,dz");
		FPrintln(file, line);
		CloseFile(file);
	}
};
