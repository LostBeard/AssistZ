namespace AssistZ.Audio
{
    /// <summary>Which side of the listener a tracked source is on, once the view has turned enough to tell.</summary>
    public enum SourceSide { Unknown, Front, Back }

    /// <summary>A track's state after an event joined it.</summary>
    public readonly record struct TrackResult(int TrackId, SourceSide Side, double Confidence, int Events, double TurnedDegrees);

    /// <summary>
    /// Resolves front from back for REPEATING sounds (footsteps, an engine, a zombie) from how their stereo azimuth moves
    /// while the view turns. A stereo mix folds the circle onto the front half: a source at true head-relative bearing b
    /// reads as fold(b) (b in front, 180 - b behind; measured in DayZ: the mix carries no other front/back cue). With a
    /// world-fixed source and the view turned by d since the first event, the reading becomes fold(b0 - d): a FRONT
    /// source's reading moves by -d, a BACK source's by +d. Each track keeps both hypotheses and scores every new event
    /// against both predictions; the smaller accumulated error wins once the view has turned
    /// <see cref="MinTurnDegrees"/> during the track (without turning both predict the same reading and nothing is known).
    /// </summary>
    public sealed class SourceTracker
    {
        /// <summary>
        /// An event joins a track when it is within this many degrees of either hypothesis' prediction. Lab (2026-11-03): two
        /// beacons whose stereo readings were 12 deg apart (+6.9 / -6.9) merged into one track at 14; measured azimuth
        /// noise is ~1-3 deg, so 8 keeps nearby sources apart.
        /// </summary>
        public double MatchDegrees { get; set; } = 8;
        /// <summary>A track ends after this long without an event.</summary>
        public TimeSpan TrackTimeout { get; set; } = TimeSpan.FromSeconds(4);
        /// <summary>Least view turn (degrees, peak-to-peak during the track) before a side is decided.</summary>
        public double MinTurnDegrees { get; set; } = 12;
        /// <summary>Mean-squared-error ratio the losing hypothesis must exceed the winner by.</summary>
        public double DecisionRatio { get; set; } = 2.5;

        sealed class Track
        {
            public int Id;
            public double BearingFront;   // world bearing if the source is in front: az + yaw at first event
            public double BearingBack;    // world bearing if behind: 180 - az + yaw
            public double ErrFront, ErrBack;
            public int Events;
            public DateTime Last;
            public double YawMin, YawMax;
            public double LastAzimuth;
        }

        readonly List<Track> _tracks = new();
        int _nextId = 1;

        /// <summary>Tracks currently alive.</summary>
        public int ActiveTracks => _tracks.Count;

        /// <summary>
        /// Adds an event heard at calibrated front-half azimuth <paramref name="azimuth"/> (-90..90, + = right) while the
        /// view heading was <paramref name="viewYaw"/> (degrees, + = right, relative is fine). Returns its track's state.
        /// </summary>
        public TrackResult Add(double azimuth, double viewYaw, DateTime time)
        {
            _tracks.RemoveAll(t => time - t.Last > TrackTimeout);

            Track? best = null;
            double bestDistance = double.MaxValue;
            foreach (var t in _tracks)
            {
                double front = Math.Abs(azimuth - PredictFront(t, viewYaw));
                double back = Math.Abs(azimuth - PredictBack(t, viewYaw));
                double d = Math.Min(front, back);
                if (d <= MatchDegrees && d < bestDistance)
                {
                    best = t;
                    bestDistance = d;
                }
            }

            if (best == null)
            {
                best = new Track
                {
                    Id = _nextId++,
                    BearingFront = azimuth + viewYaw,
                    BearingBack = 180 - azimuth + viewYaw,
                    YawMin = viewYaw,
                    YawMax = viewYaw,
                };
                _tracks.Add(best);
            }
            else
            {
                double ef = azimuth - PredictFront(best, viewYaw);
                double eb = azimuth - PredictBack(best, viewYaw);
                best.ErrFront += ef * ef;
                best.ErrBack += eb * eb;
            }
            best.Events++;
            best.Last = time;
            best.LastAzimuth = azimuth;
            best.YawMin = Math.Min(best.YawMin, viewYaw);
            best.YawMax = Math.Max(best.YawMax, viewYaw);
            return Result(best);
        }

        TrackResult Result(Track t)
        {
            double turned = t.YawMax - t.YawMin;
            var side = SourceSide.Unknown;
            double confidence = 0;
            if (turned >= MinTurnDegrees && t.Events >= 2)
            {
                // Mean squared errors with a 1 deg^2 floor so a perfect fit does not divide by zero.
                double n = t.Events - 1;
                double mf = t.ErrFront / n + 1, mb = t.ErrBack / n + 1;
                if (mb >= mf * DecisionRatio) side = SourceSide.Front;
                else if (mf >= mb * DecisionRatio) side = SourceSide.Back;
                confidence = 1 - Math.Min(mf, mb) / Math.Max(mf, mb);
            }
            return new TrackResult(t.Id, side, confidence, t.Events, turned);
        }

        static double PredictFront(Track t, double viewYaw) => Fold(t.BearingFront - viewYaw);
        static double PredictBack(Track t, double viewYaw) => Fold(t.BearingBack - viewYaw);

        /// <summary>The stereo fold: a head-relative bearing (any angle) to the front-half azimuth a stereo mix shows (-90..90).</summary>
        public static double Fold(double bearing)
        {
            double b = ((bearing + 180) % 360 + 360) % 360 - 180;   // -180..180
            if (b > 90) return 180 - b;
            if (b < -90) return -180 - b;
            return b;
        }
    }
}
