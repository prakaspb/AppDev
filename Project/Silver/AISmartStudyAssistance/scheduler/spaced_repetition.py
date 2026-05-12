from datetime import date, timedelta


def get_review_schedule(mastery, last_review_date=None):
    """
    Returns the next review date based on mastery score
    """

    if mastery < 0.4:
        days = 1
    elif mastery < 0.7:
        days = 3
    else:
        days = 7

    base_date = last_review_date or date.today()
    next_review = base_date + timedelta(days=days)

    return days, next_review
