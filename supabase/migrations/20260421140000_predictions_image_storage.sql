-- Prediction images: path in Storage + private bucket + RLS by auth.uid() prefix.

alter table public.predictions
    add column if not exists image_storage_path text;

insert into storage.buckets (id, name, public)
values ('prediction-images', 'prediction-images', false)
on conflict (id) do nothing;

drop policy if exists "prediction_images_insert_own" on storage.objects;
create policy "prediction_images_insert_own"
    on storage.objects
    for insert
    to authenticated
    with check (
        bucket_id = 'prediction-images'
        and split_part(name, '/', 1) = auth.uid()::text
    );

drop policy if exists "prediction_images_select_own" on storage.objects;
create policy "prediction_images_select_own"
    on storage.objects
    for select
    to authenticated
    using (
        bucket_id = 'prediction-images'
        and split_part(name, '/', 1) = auth.uid()::text
    );

drop policy if exists "prediction_images_update_own" on storage.objects;
create policy "prediction_images_update_own"
    on storage.objects
    for update
    to authenticated
    using (
        bucket_id = 'prediction-images'
        and split_part(name, '/', 1) = auth.uid()::text
    )
    with check (
        bucket_id = 'prediction-images'
        and split_part(name, '/', 1) = auth.uid()::text
    );

drop policy if exists "prediction_images_delete_own" on storage.objects;
create policy "prediction_images_delete_own"
    on storage.objects
    for delete
    to authenticated
    using (
        bucket_id = 'prediction-images'
        and split_part(name, '/', 1) = auth.uid()::text
    );
