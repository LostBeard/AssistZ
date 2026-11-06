namespace AssistZ.Audio
{
    /// <summary>
    /// One word per burst for world events. The probe labels each event alone, so a gas strike's first events could read "heli crash" and the HUD flickered. Events closer than
    /// <see cref="MaxGapSeconds"/> on the sample clock form a burst, and its world event words become the burst's most common one (a tie keeps the word seen first, so a settled word does not flip).
    /// Other words, gunfire above all, take no part. Simulated on the three world runs, each held out: bursts with mixed words 9/62 -> 0, gas 84 -> 91 of 97, heli 54 -> 57 of 60, no new false alarms (the five-class probe of that time); any gap
    /// from 0.5 to 4 s scored the same.
    /// One exception for gunfire, <see cref="GunfireYields"/>: thunder's sharp start reads as a shot for one event (11 of 13 false gunshots in dry storms were the first event of a thunder burst, the rest thunder), so a gunfire word outnumbered by thunder in its own burst takes thunder.
    /// It yields to no other word. A real shot within a second of a thunder burst can be lost this way: the lab's gunfire-in-a-storm runs measure that.
    /// </summary>
    public sealed class BurstVote
    {
        readonly Func<string, bool> _votes;
        readonly List<string> _order = new();
        readonly Dictionary<string, int> _count = new();
        readonly Func<string, bool> _isGunfire;
        int _gunfire;
        long _lastEnd = long.MinValue;

        public const string Thunder = "thunder";

        public BurstVote(Func<string, bool> votes, Func<string, bool>? isGunfire = null)
        {
            _votes = votes;
            _isGunfire = isGunfire ?? (_ => false);
        }

        public double MaxGapSeconds { get; set; } = 1.0;

        /// <summary>Feeds the next event, in order. Returns the burst's world event word now, or null when the burst has none. Start of the burst's first event is reported through <see cref="BurstStart"/>.</summary>
        public string? Add(long endSample, string word, double sampleRate)
        {
            if (_lastEnd == long.MinValue || endSample - _lastEnd >= MaxGapSeconds * sampleRate)
            {
                _order.Clear();
                _count.Clear();
                _gunfire = 0;
                BurstStart = endSample;
            }
            _lastEnd = endSample;
            if (_isGunfire(word)) _gunfire++;
            if (_votes(word))
            {
                if (!_count.TryGetValue(word, out int n)) _order.Add(word);
                _count[word] = n + 1;
            }
            string? best = null;
            foreach (string w in _order)
                if (best == null || _count[w] > _count[best]) best = w;
            return best;
        }

        /// <summary>True when the current burst's gunfire words are outnumbered by thunder, so they take thunder. A tie keeps the gunfire: a missed shot costs more than a false one.</summary>
        public bool GunfireYields => _gunfire > 0 && _count.TryGetValue(Thunder, out int thunder) && thunder > _gunfire;

        /// <summary>End sample of the current burst's first event.</summary>
        public long BurstStart { get; private set; }
    }
}
