<!-- LOVABLE:BEGIN -->
> [!IMPORTANT]
> This project is connected to [Lovable](https://lovable.dev). Avoid rewriting
> published git history — force pushing, or rebasing/amending/squashing commits
> that are already pushed — as it rewrites history on Lovable's side and the
> user will likely lose their project history.
>
> Commits you push to the connected branch sync back to Lovable and show up in
> the editor, so keep the branch in a working state.
<!-- LOVABLE:END -->

- Auth uses Lovable Cloud email/password via the browser client and a shared useAuth hook; every app page lives under the _authenticated layout (client-side gate, redirect to /login); only login, signup, forgot-password and reset-password are public — why: the user wants an authentication-first flow.
- Text translation runs in an authenticated server function calling the Lovable AI Gateway; speech uses the browser's speechSynthesis in its own hook — why: no Python translation endpoint exists yet and keys must stay server-side.
