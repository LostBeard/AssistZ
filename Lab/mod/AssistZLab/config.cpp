// AssistZ lab mod: client-side playback of lab events the vanilla engine does not replicate for server-controlled
// characters (a dummy survivor's gunshots are not heard on clients; DayZ-Expansion-AI also fires AI weapons outside the
// vanilla weapon FSM on the client for this reason). Loaded on BOTH the lab server and the lab client (-mod=).
class CfgPatches
{
	class AssistZLab
	{
		units[] = {};
		weapons[] = {};
		requiredVersion = 0.1;
		requiredAddons[] = { "DZ_Data", "DZ_Scripts" };
	};
};

class CfgMods
{
	class AssistZLab
	{
		dir = "AssistZLab";
		name = "AssistZLab";
		credits = "SpawnDev";
		author = "SpawnDev";
		version = "0.1";
		type = "mod";
		dependencies[] = { "Game", "World" };
		class defs
		{
			// 3_Game: the client-side bullet impact log (modded DayZGame.FirearmEffects).
			class gameScriptModule
			{
				value = "";
				files[] = { "AssistZLab/scripts/3_Game" };
			};
			class worldScriptModule
			{
				value = "";
				files[] = { "AssistZLab/scripts/4_World" };
			};
		};
	};
};
