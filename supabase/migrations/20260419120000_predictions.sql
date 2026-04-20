-- predictions + RLS (auth.uid() = user_id)
--
-- Aplicar sin copiar/pegar en el Editor:
--   1) Instala la CLI: https://supabase.com/docs/guides/cli
--   2) En la raíz del repo: `supabase login` y `supabase link --project-ref <tu-ref>`
--   3) `supabase db push`  (aplica migraciones pendientes al proyecto enlazado)
--
-- También puedes abrir este archivo y ejecutarlo en SQL Editor si prefieres.

create table if not exists public.predictions (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users (id) on delete cascade,
    risk text not null,
    score double precision not null,
    model_version text not null,
    created_at timestamptz not null default now()
);

create index if not exists predictions_user_id_idx on public.predictions (user_id);

alter table public.predictions enable row level security;

drop policy if exists "predictions_select_own" on public.predictions;
create policy "predictions_select_own"
    on public.predictions
    for select
    to authenticated
    using (auth.uid() = user_id);

drop policy if exists "predictions_insert_own" on public.predictions;
create policy "predictions_insert_own"
    on public.predictions
    for insert
    to authenticated
    with check (auth.uid() = user_id);
