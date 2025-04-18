WITH
TEST_CTE AS (
    SELECT
        Category AS category, 
        SUM(Amount) AS sum_amount 
    FROM 
        records 
    GROUP BY 
        Category 
    ORDER BY 
        sum_amount ASC 
    LIMIT 5
)

SELECT * FROM TEST_CTE;