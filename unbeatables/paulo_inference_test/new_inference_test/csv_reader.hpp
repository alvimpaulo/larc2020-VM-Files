#include <string>
#include <vector>
#include <istream>
#include <iterator>

#ifndef _CSVREADER_H_
#define _CSVREADER_H_

using namespace std;

class CSVRow
{
	private:
		string         line;
		vector<int>    data;

    public:
        string operator[](size_t index) const
        {
            return string(&line[data[index] + 1], data[index + 1] -  (data[index] + 1));
        }
        size_t size() const
        {
            return data.size() - 1;
        }
		friend istream& operator>>(istream& str, CSVRow& row)
        {
            str >> row.line;

			row.data.clear();
            row.data.push_back(-1);

            string::size_type pos = 0;
            while((pos = row.line.find(',', pos)) != string::npos)
            {
                row.data.push_back(pos);
                ++pos;
            }

            // This checks for a trailing comma with no data after it.
            pos = row.line.size();
            row.data.push_back(pos);

			return str;
		}
};

class CSVIterator
{
	public:
		typedef input_iterator_tag   iterator_category;
		typedef CSVRow                      value_type;
		typedef size_t                 difference_type;
		typedef CSVRow*                        pointer;
		typedef CSVRow&                      reference;

		CSVIterator(istream& str) : str(str.good() ? &str : NULL) { ++(*this); }
		CSVIterator()             : str(NULL) {}

		// Pre Increment
		CSVIterator& operator++()
		{
			if(str) {
				if (!((*str) >> row)) {
					str = NULL;
				}
			}
			return *this;
		}
		// Post increment
		CSVIterator operator++(int)
		{
			CSVIterator    tmp(*this);

			++(*this);
			return tmp;
		}
		CSVRow const& operator*()   const       {return row;}
		CSVRow const* operator->()  const       {return &row;}

		bool operator==(CSVIterator const& rhs) {return ((this == &rhs) || ((this->str == NULL) && (rhs.str == NULL)));}
		bool operator!=(CSVIterator const& rhs) {return !((*this) == rhs);}

	private:
		istream*       str;
		CSVRow         row;
};
#endif
