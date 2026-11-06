using Microsoft.AspNetCore.Components;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;

namespace AssistZ.Services
{
    /// <summary>
    /// Floats a page element in an always-on-top Document Picture-in-Picture window (over the game) and puts it back
    /// when that window closes. The element is MOVED, not copied: Blazor keeps its references to the nodes, so the
    /// component inside keeps rendering live. Blazor's event delegation listens on the main document, so the moved
    /// content is display-only; the controls stay on the page.
    /// </summary>
    public sealed class OverlayWindow : IDisposable
    {
        readonly SpawnJSRuntime _js;
        readonly NavigationManager _nav;
        Window? _pip;
        Element? _content, _home;

        public OverlayWindow(SpawnJSRuntime js, NavigationManager nav)
        {
            _js = js;
            _nav = nav;
        }

        public static bool Supported => SpawnJSRuntime.Instance?.Has("documentPictureInPicture") == true;
        public bool IsOpen => _pip != null;
        public event Action? Changed;

        /// <summary>Moves element <paramref name="contentId"/> into a new PiP window; it returns to <paramref name="homeId"/> on close.</summary>
        public async Task OpenAsync(string contentId, string homeId, int width, int height)
        {
            if (_pip != null) return;
            using var window = _js.Get<Window>("window");
            using var dpip = window.DocumentPictureInPicture;
            _pip = await dpip.RequestWindow(new PIPRequestWindowOptions { Width = width, Height = height });
            using (var doc = _pip.Document)
            using (var head = doc.Head!)
            using (var body = doc.Body!)
            {
                using var link = doc.CreateElement("link");
                link.SetAttribute("rel", "stylesheet");
                link.SetAttribute("href", _nav.ToAbsoluteUri("css/app.css").ToString());
                head.AppendChild(link).Dispose();
                body.SetAttribute("class", "pip");
                _content = _js.Call<string, Element>("document.getElementById", contentId);
                _home = _js.Call<string, Element>("document.getElementById", homeId);
                body.AppendChild(_content).Dispose();
            }
            _pip.OnPageHide += OnClosed;
            Changed?.Invoke();
        }

        public void Close() => _pip?.Close();

        void OnClosed(PageTransitionEvent e)
        {
            if (_home != null && _content != null) _home.AppendChild(_content).Dispose();
            _content?.Dispose();
            _home?.Dispose();
            _content = _home = null;
            if (_pip != null) _pip.OnPageHide -= OnClosed;
            _pip?.Dispose();
            _pip = null;
            Changed?.Invoke();
        }

        public void Dispose() => Close();
    }
}
