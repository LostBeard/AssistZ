namespace AssistZ.Video
{
    /// <summary>
    /// Which measured view turns to believe. Found live (2026-11-05, character standing still, game log yaw constant): the heading climbed 0 to 161 degrees and front/back came out wrong 19 times to 11, at confidence near 1. Each jump was one frame of about -245 px, at the tracker's search limit, with a good score and stillness either side. The likely cause is on-screen text changing over a flat sky, which tips the column profile. Raising the score bar was no use: the worst jump scored 0.94, while real fast turns fall as low as 0.72 and a bar of 0.8 loses 40% of them.
    /// So: a peak at the edge of the search is not a measurement, and a jump with stillness on both sides is not a turn. On that run false heading fell from 234 to 13 degrees of path, on recorded real turns about 94% of the movement stays. A real one-frame flick from rest is dropped too: that costs a few degrees, not a false front/back.
    /// A jump from rest is held one frame until the next says whether anything is moving.
    /// </summary>
    public sealed class YawGate
    {
        /// <summary>Below this the frames did not line up (a cut, a menu, a flash).</summary>
        public const float MinScore = 0.5f;
        /// <summary>At or past this the peak sits at the search's edge (<c>YawTracker.MaxShift</c> is 246).</summary>
        public const float EdgePixels = 240f;
        /// <summary>A shift this big from rest waits for the next frame.</summary>
        public const float JumpPixels = 40f;
        /// <summary>Under this the view is still.</summary>
        public const float RestPixels = 2f;

        float _previousRaw;
        float _held;
        bool _holding;

        /// <summary>The turn to integrate for this frame, in pixels: the shift, 0, or a released jump plus the shift.</summary>
        public float Next(float shiftPixels, float score)
        {
            float raw = shiftPixels;
            float usable = score > MinScore && MathF.Abs(raw) < EdgePixels ? raw : 0f;
            float result;
            if (_holding)
            {
                _holding = false;
                bool moving = MathF.Abs(raw) >= RestPixels;
                result = moving ? _held + usable : usable;
            }
            else if (MathF.Abs(usable) >= JumpPixels && MathF.Abs(_previousRaw) < RestPixels)
            {
                _held = usable;
                _holding = true;
                result = 0f;
            }
            else
            {
                result = usable;
            }
            _previousRaw = raw;
            return result;
        }
    }
}
