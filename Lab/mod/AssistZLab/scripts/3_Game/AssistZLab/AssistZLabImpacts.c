// AssistZ lab: ground truth for where every bullet LANDS, logged on the client (FirearmEffects runs on clients for every
// impact, to play its effects): game time, ammo, surface, impact position, impact speed and the distance to the local
// player's head. Separates "a bullet hit the ground 3 m away" from "a bullet flew past" when the analysis looks at the
// sharp sound before a report (2026-11-03: a > 3 kHz precursor 115-140 ms before close SVD reports - crack or impact?).
modded class DayZGame
{
	override void FirearmEffects(Object source, Object directHit, int componentIndex, string surface, vector pos, vector surfNormal, vector exitPos, vector inSpeed, vector outSpeed, bool isWater, bool deflected, string ammoType)
	{
		super.FirearmEffects(source, directHit, componentIndex, surface, pos, surfNormal, exitPos, inSpeed, outSpeed, isWater, deflected, ammoType);
		if (IsServer())
			return;
		float headDist = -1;
		Man me = GetPlayer();
		if (me)
		{
			vector head = me.GetPosition();
			head[1] = head[1] + 1.6;
			headDist = vector.Distance(head, pos);
		}
		string hit = "none";
		if (directHit)
			hit = directHit.GetType();
		bool water = isWater;
		bool deflect = deflected;
		float speed = inSpeed.Length();
		string line = GetTime().ToString() + "," + ammoType + "," + surface + "," + pos[0].ToString() + "," + pos[1].ToString() + "," + pos[2].ToString();
		line += "," + speed.ToString() + "," + headDist.ToString() + "," + hit + "," + water.ToString() + "," + deflect.ToString();
		string path = "$profile:assistz_impacts.csv";
		bool exists = FileExist(path);
		FileHandle file = OpenFile(path, FileMode.APPEND);
		if (!file)
			return;
		if (!exists)
			FPrintln(file, "game_ms,ammo,surface,x,y,z,speed,head_dist,direct_hit,water,deflected");
		FPrintln(file, line);
		CloseFile(file);
	}
};
