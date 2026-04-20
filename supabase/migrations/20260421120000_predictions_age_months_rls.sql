-- Predictions: age_months (from birth_date), drop legacy age; UPDATE RLS; ensure index.

alter table public.predictions drop column if exists age;

alter table public.predictions
    add column if not exists age_months integer;

create index if not exists predictions_user_id_idx on public.predictions (user_id);

drop policy if exists "predictions_update_own" on public.predictions;
create policy "predictions_update_own"
    on public.predictions
    for update
    to authenticated
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);
