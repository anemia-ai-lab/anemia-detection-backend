-- Sync offline metadata, historial extendido, preprocessing JSONB, delete RLS.

alter table public.predictions
    add column if not exists client_id uuid,
    add column if not exists inference_mode text not null default 'backend',
    add column if not exists raw_probability double precision,
    add column if not exists calibrated_probability double precision,
    add column if not exists threshold_used double precision,
    add column if not exists prediction smallint,
    add column if not exists client_created_at timestamptz,
    add column if not exists image_sha256 text,
    add column if not exists synced_at timestamptz,
    add column if not exists preprocessing jsonb;

-- Orden cronológico de campo: COALESCE(client_created_at, created_at)
alter table public.predictions
    add column if not exists effective_created_at timestamptz
    generated always as (coalesce(client_created_at, created_at)) stored;

create unique index if not exists predictions_user_client_id_uidx
    on public.predictions (user_id, client_id)
    where client_id is not null;

create index if not exists predictions_user_effective_created_idx
    on public.predictions (user_id, effective_created_at desc, id desc);

grant update, delete on table public.predictions to authenticated;

drop policy if exists "predictions_delete_own" on public.predictions;
create policy "predictions_delete_own"
    on public.predictions
    for delete
    to authenticated
    using (auth.uid() = user_id);
