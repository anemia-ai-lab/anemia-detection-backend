import logging

from postgrest import APIError

from backend.core.exceptions import AuthServiceError
from backend.repositories.profiles_repository import ProfilesRepository
from backend.schemas.auth import UserOut
from backend.schemas.profile import ProfileOut, ProfilePatchRequest

logger = logging.getLogger(__name__)


def _names_complete(first: str | None, last: str | None) -> bool:
    return bool((first or "").strip() and (last or "").strip())


class ProfileService:
    def __init__(self, repo: ProfilesRepository | None = None) -> None:
        self._repo = repo or ProfilesRepository()

    def get_profile(self, user: UserOut, access_token: str) -> ProfileOut:
        try:
            row = self._repo.fetch_by_user_id(access_token, user.id)
        except APIError as e:
            logger.warning(
                "profile_db_error op=fetch code=%s message=%s",
                e.code or "profile_fetch_failed",
                e.message or "fetch_failed",
            )
            raise AuthServiceError(
                "Could not load profile",
                502,
                code=e.code or "profile_fetch_failed",
            ) from e
        if row is None:
            return ProfileOut(
                id=user.id,
                email=user.email,
                has_profile_row=False,
                profile_completed=False,
                first_name=None,
                last_name=None,
                department=None,
                province=None,
                created_at=None,
            )
        return ProfileOut(
            id=str(row["id"]),
            email=user.email,
            has_profile_row=True,
            profile_completed=bool(row.get("profile_completed", False)),
            first_name=row.get("first_name"),
            last_name=row.get("last_name"),
            department=row.get("department"),
            province=row.get("province"),
            created_at=row.get("created_at"),
        )

    def update_profile(
        self,
        user: UserOut,
        access_token: str,
        patch: ProfilePatchRequest,
    ) -> ProfileOut:
        try:
            row = self._repo.fetch_by_user_id(access_token, user.id)
        except APIError as e:
            logger.warning(
                "profile_db_error op=fetch_for_update code=%s message=%s",
                e.code or "profile_fetch_failed",
                e.message or "fetch_failed",
            )
            raise AuthServiceError(
                "Could not load profile",
                502,
                code=e.code or "profile_fetch_failed",
            ) from e
        cur = row or {}
        data = patch.model_dump(exclude_unset=True)
        first = data["first_name"] if "first_name" in data else cur.get("first_name")
        last = data["last_name"] if "last_name" in data else cur.get("last_name")
        dept = data["department"] if "department" in data else cur.get("department")
        prov = data["province"] if "province" in data else cur.get("province")
        completed = _names_complete(first, last)
        payload = {
            "first_name": first,
            "last_name": last,
            "department": dept,
            "province": prov,
            "profile_completed": completed,
        }
        try:
            self._repo.upsert_profile(access_token, user_id=user.id, payload=payload)
        except APIError as e:
            logger.warning(
                "profile_db_error op=upsert code=%s message=%s",
                e.code or "profile_save_failed",
                e.message or "save_failed",
            )
            raise AuthServiceError(
                "Could not save profile",
                502,
                code=e.code or "profile_save_failed",
            ) from e
        return self.get_profile(user, access_token)
